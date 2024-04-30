//! py-binding for the VJ model

use anyhow::{anyhow, Result};
use ndarray::{s, Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use righor::shared::{utils::Normalize, utils::Normalize2, ResultInference};
use righor::vdj::{display_j_alignment, display_v_alignment, Sequence};
use righor::vj::Generator;
use righor::vj::Model;
use righor::{Dna, Gene, Modelable};
use std::fs;
use std::path::Path;

#[pyclass(name = "Model")]
#[derive(Debug, Clone)]
pub struct PyModel {
    inner: righor::vj::Model,
}

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
        Ok(PyModel { inner: m })
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
        Ok(PyModel { inner: m })
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
            inner: righor::vj::Model::load_json(&path)?,
        })
    }

    fn __deepcopy__(&self, _memo: &PyDict) -> Self {
        self.clone()
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    #[pyo3(name = "display_v_alignment")]
    pub fn py_display_v_alignment(
        &self,
        seq: &str,
        v_al: &righor::VJAlignment,
        align_params: &righor::AlignmentParameters,
    ) -> Result<String> {
        let seq_dna = righor::Dna::from_string(seq)?;
        Ok(display_v_alignment(
            &seq_dna,
            v_al,
            &self.inner.inner,
            align_params,
        ))
    }

    #[pyo3(name = "display_j_alignment")]
    pub fn py_display_j_alignment(
        &self,
        seq: &str,
        j_al: &righor::VJAlignment,
        align_params: &righor::AlignmentParameters,
    ) -> Result<String> {
        let seq_dna = righor::Dna::from_string(seq)?;
        Ok(display_j_alignment(
            &seq_dna,
            j_al,
            &self.inner.inner,
            align_params,
        ))
    }

    pub fn generator(
        &self,
        seed: Option<u64>,
        available_v: Option<Vec<Gene>>,
        available_j: Option<Vec<Gene>>,
    ) -> Result<Generator> {
        Generator::new(self.inner.clone(), seed, available_v, available_j)
    }

    /// Return an uniform (blank slate) model with the same v/d/j genes
    /// as the current model.
    pub fn uniform(&self) -> PyResult<PyModel> {
        Ok(PyModel {
            inner: self.inner.uniform()?,
        })
    }

    #[pyo3(signature = (dna_seq, align_params=righor::AlignmentParameters::default()))]
    /// Align one nucleotide sequence and return a `Sequence` object
    pub fn align_sequence(
        &self,
        dna_seq: &str,
        align_params: righor::AlignmentParameters,
    ) -> Result<Sequence> {
        let dna = righor::Dna::from_string(dna_seq)?;
        let alignment = self.inner.align_sequence(&dna, &align_params)?;
        Ok(alignment)
    }

    pub fn align_and_infer(
        &mut self,
        str_seqs: Vec<String>,
        align_params: &righor::AlignmentParameters,
        inference_params: &righor::InferenceParameters,
    ) -> Result<()> {
        let dna_seqs = str_seqs
            .iter()
            .map(|x| righor::Dna::from_string(x))
            .collect::<Result<Vec<_>>>()?;
        self.inner
            .align_and_infer(&dna_seqs, align_params, inference_params)?;
        Ok(())
    }

    #[pyo3(signature = (cdr3_seqs, inference_params=righor::InferenceParameters::default()))]
    pub fn align_and_infer_from_cdr3(
        &mut self,
        cdr3_seqs: Vec<(String, Vec<Gene>, Vec<Gene>)>,

        inference_params: righor::InferenceParameters,
    ) -> Result<()> {
        let dna_seqs = cdr3_seqs
            .iter()
            .map(|(x, v, j)| (righor::Dna::from_string(x).unwrap(), v.clone(), j.clone()))
            .collect::<Vec<_>>();
        self.inner
            .align_and_infer_from_cdr3(&dna_seqs, &inference_params)?;
        Ok(())
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
        align_params: &righor::AlignmentParameters,
    ) -> Result<Vec<Sequence>> {
        dna_seqs
            .par_iter()
            .map(|seq| {
                let dna = righor::Dna::from_string(seq)?;
                let alignment = self.inner.align_sequence(&dna, align_params)?;
                Ok(alignment)
            })
            .collect()
    }

    #[pyo3(signature = (sequence, align_params=righor::AlignmentParameters::default_evaluate(), infer_params=righor::InferenceParameters::default_evaluate()))]
    /// Evaluate the sequence and return the most likely recombination scenario
    /// as well as its probability of being generated.
    pub fn evaluate(
        &self,
        sequence: String,
        align_params: righor::AlignmentParameters,
        infer_params: righor::InferenceParameters,
    ) -> Result<ResultInference> {
        let al = self.align_sequence(&sequence, align_params)?;
        Ok(self.inner.evaluate(&al, &infer_params)?)
    }

    #[pyo3(signature = (sequences, inference_params=righor::InferenceParameters::default()))]
    /// Run one round of expectation-maximization on the current model and update it.
    pub fn infer(
        &mut self,
        sequences: Vec<Sequence>,
        inference_params: righor::InferenceParameters,
    ) -> Result<()> {
        let alignments = sequences.into_iter().map(|s| s).collect::<Vec<Sequence>>();
        let mut model = self.inner.clone();
        model.infer(&alignments, &inference_params)?;
        self.inner = model.clone();
        Ok(())
    }

    #[getter]
    fn get_v_segments(&self) -> Vec<Gene> {
        self.inner.seg_vs.to_owned()
    }

    #[setter]
    /// Update the v segments and adapt the associated marginals
    fn set_v_segments(&mut self, value: Vec<Gene>) -> Result<()> {
        let [_, sj] = *self.inner.get_p_vj().shape() else {
            return Err(anyhow!("Something is wrong with the v segments"));
        };
        let mut new_p_vj = Array2::<f64>::zeros([value.len(), sj]);

        let [sdelv, _] = *self.inner.p_del_v_given_v.shape() else {
            return Err(anyhow!("Something is wrong with the v segments"));
        };
        let mut new_p_del_v_given_v = Array2::<f64>::zeros([sdelv, value.len()]);

        let proba_v_default = 1. / (value.len() as f64);
        let delv_default =
            self.inner.p_del_v_given_v.sum_axis(Axis(1)) / self.inner.p_del_v_given_v.sum();

        for (iv, v) in value.iter().enumerate() {
            match self
                .inner
                .seg_vs
                .iter()
                .enumerate()
                .find(|(_index, g)| g.name == v.name)
            {
                Some((index, _gene)) => {
                    new_p_vj
                        .slice_mut(s![iv, ..])
                        .assign(&self.inner.get_p_vj().slice_mut(s![index, ..]));
                    new_p_del_v_given_v
                        .slice_mut(s![.., iv])
                        .assign(&self.inner.p_del_v_given_v.slice_mut(s![.., index]));
                }
                None => {
                    new_p_vj.slice_mut(s![iv, ..]).fill(proba_v_default);
                    new_p_del_v_given_v
                        .slice_mut(s![.., iv])
                        .assign(&delv_default);
                }
            }
        }

        // normalzie
        new_p_vj = new_p_vj.normalize_distribution_double()?;
        new_p_del_v_given_v = new_p_del_v_given_v.normalize_distribution()?;

        self.inner.seg_vs = value;
        self.inner.set_p_vj(&new_p_vj)?;
        self.inner.p_del_v_given_v = new_p_del_v_given_v;
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_j_segments(&self) -> Vec<Gene> {
        self.inner.seg_js.to_owned()
    }

    #[setter]
    /// Update the j segments and adapt the associated marginals
    fn set_j_segments(&mut self, value: Vec<Gene>) -> Result<()> {
        let [sv, _] = *self.inner.get_p_vj().shape() else {
            return Err(anyhow!("Something is wrong with the j segments"));
        };
        let mut new_p_vj = Array2::<f64>::zeros([sv, value.len()]);

        let [sdelj, _] = *self.inner.p_del_j_given_j.shape() else {
            return Err(anyhow!("Something is wrong with the j segments"));
        };
        let mut new_p_del_j_given_j = Array2::<f64>::zeros([sdelj, value.len()]);

        let proba_j_default = 1. / (value.len() as f64);
        let delj_default =
            self.inner.p_del_j_given_j.sum_axis(Axis(1)) / self.inner.p_del_j_given_j.sum();

        for (ij, j) in value.iter().enumerate() {
            match self
                .inner
                .seg_js
                .iter()
                .enumerate()
                .find(|(_index, g)| g.name == j.name)
            {
                Some((index, _gene)) => {
                    new_p_vj
                        .slice_mut(s![.., ij])
                        .assign(&self.inner.get_p_vj().slice_mut(s![.., index]));
                    new_p_del_j_given_j
                        .slice_mut(s![.., ij])
                        .assign(&self.inner.p_del_j_given_j.slice_mut(s![.., index]));
                }
                None => {
                    new_p_vj.slice_mut(s![.., ij]).fill(proba_j_default);
                    new_p_del_j_given_j
                        .slice_mut(s![.., ij])
                        .assign(&delj_default);
                }
            }
        }

        // normalzie
        new_p_vj = new_p_vj.normalize_distribution_double()?;
        new_p_del_j_given_j = new_p_del_j_given_j.normalize_distribution()?;

        self.inner.seg_js = value;
        self.inner.set_p_vj(&new_p_vj)?;
        self.inner.p_del_j_given_j = new_p_del_j_given_j;
        self.inner.initialize()?;
        Ok(())
    }

    #[setter]
    fn set_range_del_v(&mut self, value: (i64, i64)) -> PyResult<()> {
        self.inner.range_del_v = value;
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_range_del_v(&self) -> (i64, i64) {
        self.inner.range_del_v
    }

    #[setter]
    fn set_range_del_j(&mut self, value: (i64, i64)) -> PyResult<()> {
        self.inner.range_del_j = value;
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_range_del_j(&self) -> (i64, i64) {
        self.inner.range_del_j
    }

    #[getter]
    fn get_p_v(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.p_v.to_owned().into_pyarray(py).to_owned()
    }

    /// Return the marginal on (D, J)
    #[getter]
    fn get_p_j(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.get_p_j().to_owned().into_pyarray(py).to_owned()
    }

    #[getter(p_vj)]
    fn get_p_vj(&self, py: Python) -> Py<PyArray2<f64>> {
        self.inner.get_p_vj().to_owned().into_pyarray(py).to_owned()
    }

    #[setter(p_vj)]
    fn set_p_vj(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.inner.set_p_vj(&value.as_ref(py).to_owned_array())?;
        Ok(())
    }

    #[getter]
    fn get_p_ins_vj(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.p_ins_vj.to_owned().into_pyarray(py).to_owned()
    }

    #[setter]
    fn set_p_ins_vj(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.inner.p_ins_vj = value.as_ref(py).to_owned_array();
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_p_del_v_given_v(&self, py: Python) -> Py<PyArray2<f64>> {
        self.inner
            .p_del_v_given_v
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }

    #[setter]
    fn set_p_del_v_given_v(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.inner.p_del_v_given_v = value.as_ref(py).to_owned_array();
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_p_del_j_given_j(&self, py: Python) -> Py<PyArray2<f64>> {
        self.inner
            .p_del_j_given_j
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }

    #[setter]
    fn set_p_del_j_given_j(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.inner.p_del_j_given_j = value.as_ref(py).to_owned_array();
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_markov_coefficients_vj(&self, py: Python) -> Py<PyArray2<f64>> {
        self.inner
            .markov_coefficients_vj
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }

    #[setter]
    fn set_markov_coefficients_vj(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.inner.markov_coefficients_vj = value.as_ref(py).to_owned_array();
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_first_nt_bias_ins_vj(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner
            .first_nt_bias_ins_vj
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }

    #[getter]
    fn get_error_rate(&self) -> f64 {
        self.inner.error_rate
    }

    #[setter]
    fn set_error_rate(&mut self, value: f64) -> Result<()> {
        self.inner.error_rate = value;
        self.inner.initialize()?;
        Ok(())
    }
}
