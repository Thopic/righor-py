use anyhow::{anyhow, Result};
use numpy::IntoPyArray;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use righor::shared::model::*;
use righor::shared::VJAlignment;
use righor::shared::{errors::PyErrorParameters, Features};
use righor::shared::{Dna, Gene};
use righor::vdj::model::EntrySequence;
use righor::vdj::Sequence;

use rayon::prelude::*;

use std::fs;
use std::path::Path;

pub use righor::{AlignmentParameters, InferenceParameters};

#[pyclass(name = "Model")]
#[derive(Debug, Clone)]
pub struct PyModel {
    inner: Model,
    // currently analyzed features
    features: Option<Vec<Features>>,
}

// Just a copy paste of the previous impl block (I know it's shit, but pyo3 cfg_attr are a mess)
#[pymethods]
impl PyModel {
    #[staticmethod]
    /// Load the model based on species/chain/id names and location (model_dir)
    pub fn load_model(
        species: &str,
        chain: &str,
        model_dir: &str,
        id: Option<String>,
    ) -> Result<PyModel> {
        let m = Model::load_from_name(species, chain, id, Path::new(model_dir))?;
        Ok(PyModel {
            inner: m,
            features: None,
        })
    }

    #[staticmethod]
    /// Load the model from an igor-format save
    pub fn load_model_from_files(
        path_params: &str,
        path_marginals: &str,
        path_anchor_vgene: &str,
        path_anchor_jgene: &str,
    ) -> Result<PyModel> {
        let m = Model::load_from_files(
            Path::new(path_params),
            Path::new(path_marginals),
            Path::new(path_anchor_vgene),
            Path::new(path_anchor_jgene),
        )?;
        Ok(PyModel {
            inner: m,
            features: None,
        })
    }

    /// Save the model in IGoR format: create a directory 'directory'
    /// and four files that represents the model.
    pub fn save_model(&self, directory: &str) -> Result<()> {
        let path = Path::new(directory);
        match fs::create_dir(path) {
            Ok(_) => self.inner.save_model(path),
            Err(e) => Err(e.into()),
        }
    }

    /// Save the model in json format
    pub fn save_json(&self, filename: &str) -> Result<()> {
        let path = Path::new(filename);
        self.inner.save_json(&path)
    }

    /// Save the model in json format
    #[staticmethod]
    pub fn load_json(filename: &str) -> Result<PyModel> {
        let path = Path::new(filename);
        Ok(PyModel {
            inner: righor::shared::Model::load_json(&path)?,
            features: None,
        })
    }

    fn __deepcopy__(&self, _memo: &PyDict) -> Self {
        self.clone()
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    pub fn generator(
        &self,
        seed: Option<u64>,
        available_v: Option<Vec<Gene>>,
        available_j: Option<Vec<Gene>>,
    ) -> Result<Generator> {
        Generator::new(self.inner.clone(), seed, available_v, available_j)
    }

    pub fn filter_vs(&self, vs: Vec<Gene>) -> Result<PyModel> {
        Ok(PyModel {
            inner: self.inner.filter_vs(vs)?,
            features: None,
        })
    }

    pub fn filter_js(&self, js: Vec<Gene>) -> Result<PyModel> {
        Ok(PyModel {
            inner: self.inner.filter_js(js)?,
            features: None,
        })
    }

    /// Return an uniform (blank slate) model with the same v/d/j genes
    /// as the current model.
    pub fn uniform(&self) -> Result<PyModel> {
        Ok(PyModel {
            inner: self.inner.uniform()?,
            features: None,
        })
    }

    /// Update the internal state of the model so it stays consistent
    fn initialize(&mut self) -> Result<()> {
        match &mut self.inner {
            Model::VDJ(x) => x.initialize(),
            Model::VJ(x) => x.initialize(),
        }
    }

    #[pyo3(signature = (seqs, align_params=righor::shared::AlignmentParameters::default_evaluate(), inference_params=righor::shared::InferenceParameters::default_evaluate()))]
    /// Infer the model. str_seqs can be either a list of aligned sequences, a list of nucleotide sequences or a list of (cdr3, V, J) sequences.
    pub fn infer(
        &mut self,
        seqs: &PyAny,
        align_params: righor::shared::AlignmentParameters,
        inference_params: righor::shared::InferenceParameters,
    ) -> Result<()> {
        let opt_sequences: Result<Vec<EntrySequence>, _> = (|| {
            if let Ok(seq) = seqs.extract::<Vec<Sequence>>() {
                return seq
                    .into_iter()
                    .map(|x| Ok(EntrySequence::Aligned(x)))
                    .collect::<Result<Vec<_>>>();
            }
            if let Ok(seq) = seqs.extract::<Vec<String>>() {
                return seq
                    .into_iter()
                    .map(|x| Ok(EntrySequence::NucleotideSequence(Dna::from_string(&x)?)))
                    .collect::<Result<Vec<_>>>();
            }
            if let Ok(seq) = seqs.extract::<Vec<(String, Vec<Gene>, Vec<Gene>)>>() {
                return seq
                    .into_iter()
                    .map(|(x, v, j)| {
                        Ok(EntrySequence::NucleotideCDR3((Dna::from_string(&x)?, v, j)))
                    })
                    .collect::<Result<Vec<_>>>();
            }
            Err(anyhow!("The sequences do not match any known types, available types are `Sequence`, `str` and `(str, [Gene], [Gene])`"))
        })();

        let sequences = opt_sequences?;

        self.features = Some(self.inner.infer(
            &sequences,
            self.features.clone(),
            &align_params,
            &inference_params,
        )?);

        Ok(())
    }

    /// Align one nucleotide sequence and return a `Sequence` object
    pub fn align_sequence(
        &self,
        seq: &str,
        align_params: &AlignmentParameters,
    ) -> Result<Sequence> {
        let dna = righor::shared::Dna::from_string(seq)?;
        let alignment = self.inner.align_sequence(&dna, align_params)?;
        Ok(alignment)
    }

    /// Given a cdr3 sequence + V/J genes return a `Sequence` object
    pub fn align_cdr3(
        &self,
        cdr3_seq: Dna,
        vgenes: Vec<Gene>,
        jgenes: Vec<Gene>,
    ) -> Result<Sequence> {
        self.inner.align_from_cdr3(&cdr3_seq, &vgenes, &jgenes)
    }

    /// Align multiple sequences (parallelized, so a bit faster than individual alignment)
    pub fn align_all_sequences(
        &self,
        dna_seqs: Vec<String>,
        align_params: &AlignmentParameters,
    ) -> Result<Vec<Sequence>> {
        dna_seqs
            .par_iter()
            .map(|seq| {
                let dna = righor::shared::Dna::from_string(seq)?;
                let alignment = self.inner.align_sequence(&dna, align_params)?;
                Ok(alignment)
            })
            .collect()
    }

    #[pyo3(signature = (sequence, align_params=righor::shared::AlignmentParameters::default_evaluate(), infer_params=righor::shared::InferenceParameters::default_evaluate()))]
    /// Evaluate the sequence and return the most likely recombination scenario
    /// as well as its probability of being generated.
    pub fn evaluate(
        &self,
        py: Python,
        sequence: &PyAny,
        align_params: righor::shared::AlignmentParameters,
        infer_params: righor::shared::InferenceParameters,
    ) -> Result<PyObject> {
        let opt_esequence: Result<EntrySequence> = (|| {
            if let Ok(s) = sequence.extract::<Sequence>() {
                return Ok(EntrySequence::Aligned(s));
            }
            if let Ok(s) = sequence.extract::<String>() {
                return Ok(EntrySequence::NucleotideSequence(Dna::from_string(&s)?));
            }
            if let Ok((s, v, j)) = sequence.extract::<(String, Vec<Gene>, Vec<Gene>)>() {
                return Ok(EntrySequence::NucleotideCDR3((Dna::from_string(&s)?, v, j)));
            }
            Err(anyhow!("The sequence does not match any known types, available types are `Sequence`, `str` and `(str, [Gene], [Gene])`"))
        })();

        if opt_esequence.is_ok() {
            let esequence = opt_esequence?;
            return Ok(self
                .inner
                .evaluate(esequence, &align_params, &infer_params)?
                .into_py(py));
        }
        let opt_esequence_vec: Result<Vec<EntrySequence>> = (|| {
            if let Ok(seq) = sequence.extract::<Vec<Sequence>>() {
                return seq
                    .into_iter()
                    .map(|x| Ok(EntrySequence::Aligned(x)))
                    .collect::<Result<Vec<_>>>();
            }
            if let Ok(seq) = sequence.extract::<Vec<String>>() {
                return seq
                    .into_iter()
                    .map(|x| Ok(EntrySequence::NucleotideSequence(Dna::from_string(&x)?)))
                    .collect::<Result<Vec<_>>>();
            }
            if let Ok(seq) = sequence.extract::<Vec<(String, Vec<Gene>, Vec<Gene>)>>() {
                return seq
                    .into_iter()
                    .map(|(x, v, j)| {
                        Ok(EntrySequence::NucleotideCDR3((Dna::from_string(&x)?, v, j)))
                    })
                    .collect::<Result<Vec<_>>>();
            }
            Err(anyhow!("The sequence does not match any known types, available types are `Sequence`, `str` and `(str, [Gene], [Gene])` or list of these"))
        })();

        let vec_sequences = opt_esequence_vec?;
        Ok(vec_sequences
            .into_par_iter()
            .map(|seq| self.inner.evaluate(seq, &align_params, &infer_params))
            .collect::<Result<Vec<_>>>()?
            .into_py(py))
    }

    /// Recreate the full sequence from the CDR3/vgene/jgene
    pub fn recreate_full_sequence(&self, dna_cdr3: &Dna, vgene: &Gene, jgene: &Gene) -> Dna {
        match &self.inner {
            Model::VDJ(x) => x.recreate_full_sequence(dna_cdr3, vgene, jgene),
            Model::VJ(x) => x.recreate_full_sequence(dna_cdr3, vgene, jgene),
        }
    }

    /// Test if self is similar to another model
    pub fn similar_to(&self, m: &PyModel) -> bool {
        match (&self.inner, &m.inner) {
            (Model::VDJ(x), Model::VDJ(y)) => x.similar_to(y.clone()),
            (Model::VJ(x), Model::VJ(y)) => x.similar_to(y.clone()),
            _ => false,
        }
    }

    #[staticmethod]
    pub fn display_v_alignment(
        seq: &str,
        v_al: &VJAlignment,
        model: &PyModel,
        align_params: &AlignmentParameters,
    ) -> Result<String> {
        Ok(Model::display_v_alignment(
            &Dna::from_string(seq)?,
            v_al,
            &model.inner,
            align_params,
        ))
    }

    #[staticmethod]
    pub fn display_j_alignment(
        seq: &str,
        j_al: &VJAlignment,
        model: &PyModel,
        align_params: &AlignmentParameters,
    ) -> Result<String> {
        Ok(Model::display_j_alignment(
            &Dna::from_string(seq)?,
            j_al,
            &model.inner,
            align_params,
        ))
    }

    #[setter]
    pub fn set_j_segments(&mut self, value: Vec<Gene>) -> Result<()> {
        self.inner.set_j_segments(value)
    }

    #[setter]
    pub fn set_v_segments(&mut self, value: Vec<Gene>) -> Result<()> {
        self.inner.set_v_segments(value)
    }
    #[getter]
    pub fn get_v_segments(&self) -> Vec<Gene> {
        self.inner.get_v_segments()
    }
    #[getter]
    pub fn get_j_segments(&self) -> Vec<Gene> {
        self.inner.get_j_segments()
    }
    #[getter]
    pub fn get_model_type(&self) -> ModelStructure {
        self.inner.get_model_type()
    }
    #[setter]
    pub fn set_model_type(&mut self, value: ModelStructure) -> Result<()> {
        self.inner.set_model_type(value)
    }
    #[getter]
    pub fn get_error(&self) -> PyErrorParameters {
        PyErrorParameters {
            s: self.inner.get_error(),
        }
    }
    #[setter]
    pub fn set_error(&mut self, value: PyErrorParameters) -> Result<()> {
        self.inner.set_error(value.s)
    }
    #[getter]
    pub fn get_d_segments(&self) -> Result<Vec<Gene>> {
        self.inner.get_d_segments()
    }
    #[setter]
    pub fn set_d_segments(&mut self, value: Vec<Gene>) -> Result<()> {
        self.inner.set_d_segments(value)
    }
    #[getter]
    pub fn get_p_v(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.get_p_v().to_owned().into_pyarray(py).to_owned()
    }
    #[getter]
    pub fn get_p_vdj(&self, py: Python) -> Result<Py<PyArray3<f64>>> {
        Ok(self
            .inner
            .get_p_vdj()?
            .to_owned()
            .into_pyarray(py)
            .to_owned())
    }
    #[setter]
    pub fn set_p_vdj(&mut self, py: Python, value: Py<PyArray3<f64>>) -> Result<()> {
        self.inner.set_p_vdj(value.as_ref(py).to_owned_array())
    }
    #[getter]
    pub fn get_p_ins_vd(&self, py: Python) -> Result<Py<PyArray1<f64>>> {
        Ok(self
            .inner
            .get_p_ins_vd()?
            .to_owned()
            .into_pyarray(py)
            .to_owned())
    }
    #[getter]
    pub fn get_p_ins_dj(&self, py: Python) -> Result<Py<PyArray1<f64>>> {
        Ok(self
            .inner
            .get_p_ins_dj()?
            .to_owned()
            .into_pyarray(py)
            .to_owned())
    }
    #[getter]
    pub fn get_p_ins_vj(&self, py: Python) -> Result<Py<PyArray1<f64>>> {
        Ok(self
            .inner
            .get_p_ins_vj()?
            .to_owned()
            .into_pyarray(py)
            .to_owned())
    }
    #[getter]
    pub fn get_p_del_v_given_v(&self, py: Python) -> Py<PyArray2<f64>> {
        self.inner
            .get_p_del_v_given_v()
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[setter]
    pub fn set_range_del_v(&mut self, value: (i64, i64)) -> Result<()> {
        self.inner.set_range_del_v(value)
    }
    #[getter]
    pub fn get_range_del_v(&self) -> (i64, i64) {
        self.inner.get_range_del_v()
    }
    #[setter]
    pub fn set_range_del_j(&mut self, value: (i64, i64)) -> Result<()> {
        self.inner.set_range_del_j(value)
    }
    #[getter]
    pub fn get_range_del_j(&self) -> (i64, i64) {
        self.inner.get_range_del_j()
    }
    #[setter]
    pub fn set_range_del_d3(&mut self, value: (i64, i64)) -> Result<()> {
        self.inner.set_range_del_d3(value)
    }
    #[getter]
    pub fn get_range_del_d3(&self) -> Result<(i64, i64)> {
        self.inner.get_range_del_d3()
    }
    #[setter]
    pub fn set_range_del_d5(&mut self, value: (i64, i64)) -> Result<()> {
        self.inner.set_range_del_d5(value)
    }
    #[getter]
    pub fn get_range_del_d5(&self) -> Result<(i64, i64)> {
        self.inner.get_range_del_d5()
    }
    #[setter]
    pub fn set_p_del_v_given_v(&mut self, py: Python, value: Py<PyArray2<f64>>) -> Result<()> {
        self.inner
            .set_p_del_v_given_v(value.as_ref(py).to_owned_array())
    }
    #[getter]
    pub fn get_p_del_j_given_j(&self, py: Python) -> Py<PyArray2<f64>> {
        self.inner
            .get_p_del_j_given_j()
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[setter]
    pub fn set_p_del_j_given_j(&mut self, py: Python, value: Py<PyArray2<f64>>) -> Result<()> {
        self.inner
            .set_p_del_j_given_j(value.as_ref(py).to_owned_array())
    }
    #[getter]
    pub fn get_p_del_d5_del_d3(&self, py: Python) -> Result<Py<PyArray3<f64>>> {
        Ok(self
            .inner
            .get_p_del_d5_del_d3()?
            .to_owned()
            .into_pyarray(py)
            .to_owned())
    }
    #[setter]
    pub fn set_p_del_d5_del_d3(&mut self, py: Python, value: Py<PyArray2<f64>>) -> Result<()> {
        self.inner
            .set_p_del_d5_del_d3(value.as_ref(py).to_owned_array())
    }
    #[getter]
    pub fn get_markov_coefficients_vd(&self, py: Python) -> Result<Py<PyArray2<f64>>> {
        Ok(self
            .inner
            .get_markov_coefficients_vd()?
            .to_owned()
            .into_pyarray(py)
            .to_owned())
    }
    #[setter]
    pub fn set_markov_coefficients_vd(
        &mut self,
        py: Python,
        value: Py<PyArray2<f64>>,
    ) -> Result<()> {
        self.inner
            .set_markov_coefficients_vd(value.as_ref(py).to_owned_array())
    }
    #[getter]
    pub fn get_markov_coefficients_dj(&self, py: Python) -> Result<Py<PyArray2<f64>>> {
        Ok(self
            .inner
            .get_markov_coefficients_dj()?
            .to_owned()
            .into_pyarray(py)
            .to_owned())
    }
    #[setter]
    pub fn set_markov_coefficients_dj(
        &mut self,
        py: Python,
        value: Py<PyArray2<f64>>,
    ) -> Result<()> {
        self.inner
            .set_markov_coefficients_dj(value.as_ref(py).to_owned_array())
    }
    #[getter]
    pub fn get_markov_coefficients_vj(&self, py: Python) -> Result<Py<PyArray2<f64>>> {
        Ok(self
            .inner
            .get_markov_coefficients_vj()?
            .to_owned()
            .into_pyarray(py)
            .to_owned())
    }
    #[setter]
    pub fn set_markov_coefficients_vj(
        &mut self,
        py: Python,
        value: Py<PyArray2<f64>>,
    ) -> Result<()> {
        self.inner
            .set_markov_coefficients_vj(value.as_ref(py).to_owned_array())
    }
    #[getter]
    pub fn get_first_nt_bias_ins_vj(&self, py: Python) -> Result<Py<PyArray1<f64>>> {
        Ok(self
            .inner
            .get_first_nt_bias_ins_vj()?
            .to_owned()
            .into_pyarray(py)
            .to_owned())
    }
    #[getter]
    pub fn get_first_nt_bias_ins_vd(&self, py: Python) -> Result<Py<PyArray1<f64>>> {
        Ok(self
            .inner
            .get_first_nt_bias_ins_vd()?
            .to_owned()
            .into_pyarray(py)
            .to_owned())
    }
    #[getter]
    pub fn get_first_nt_bias_ins_dj(&self, py: Python) -> Result<Py<PyArray1<f64>>> {
        Ok(self
            .inner
            .get_first_nt_bias_ins_dj()?
            .to_owned()
            .into_pyarray(py)
            .to_owned())
    }
    #[getter]
    /// Return the marginal on (D, J)
    pub fn get_p_dj(&self, py: Python) -> Result<Py<PyArray2<f64>>> {
        Ok(self
            .inner
            .get_p_dj()?
            .to_owned()
            .into_pyarray(py)
            .to_owned())
    }
}

#[pymodule]
#[pyo3(name = "_righor")]
fn righor_py(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<righor::shared::GenerationResult>()?;
    m.add_class::<righor::vdj::Sequence>()?;
    m.add_class::<righor::shared::errors::PyErrorParameters>()?;
    m.add_class::<righor::Gene>()?;
    m.add_class::<righor::Dna>()?;
    m.add_class::<righor::AminoAcid>()?;
    m.add_class::<InferenceParameters>()?;
    m.add_class::<AlignmentParameters>()?;
    Ok(())
}
