//! py-binding for the VJ model

use anyhow::Result;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use righor::vdj::{display_j_alignment, display_v_alignment, ResultInference, Sequence};
use righor::vj::Generator;
use righor::vj::Model;
use righor::Gene;
use std::path::Path;

#[pyclass(name = "Model")]
#[derive(Debug, Clone)]
pub struct PyModel {
    inner: righor::vj::Model,
}

#[pymethods]
impl PyModel {
    #[staticmethod]
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

    pub fn uniform(&self) -> PyResult<PyModel> {
        Ok(PyModel {
            inner: self.inner.uniform()?,
        })
    }

    pub fn align_sequence(
        &self,
        dna_seq: &str,
        align_params: &righor::AlignmentParameters,
    ) -> Result<Sequence> {
        let dna = righor::Dna::from_string(dna_seq)?;
        let alignment = self.inner.align_sequence(dna, align_params)?;
        Ok(alignment)
    }

    pub fn align_all_sequences(
        &self,
        dna_seqs: Vec<String>,
        align_params: &righor::AlignmentParameters,
    ) -> Result<Vec<Sequence>> {
        dna_seqs
            .par_iter()
            .map(|seq| {
                let dna = righor::Dna::from_string(seq)?;
                let alignment = self.inner.align_sequence(dna, align_params)?;
                Ok(alignment)
            })
            .collect()
    }

    pub fn evaluate(
        &self,
        sequence: &Sequence,
        inference_params: &righor::InferenceParameters,
    ) -> Result<ResultInference> {
        Ok(self.inner.evaluate(&sequence, inference_params)?)
    }

    pub fn infer(
        &mut self,
        sequences: Vec<Sequence>,
        inference_params: &righor::InferenceParameters,
    ) -> Result<()> {
        let alignments = sequences.into_iter().map(|s| s).collect();
        let mut model = self.inner.clone();
        model.infer(&alignments, inference_params)?;
        self.inner = model.clone();
        Ok(())
    }

    #[getter]
    fn get_v_segments(&self) -> Vec<Gene> {
        self.inner.seg_vs.to_owned()
    }

    #[setter]
    fn set_v_segments(&mut self, value: Vec<Gene>) -> Result<()> {
        self.inner.seg_vs = value;
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_j_segments(&self) -> Vec<Gene> {
        self.inner.seg_js.to_owned()
    }

    #[setter]
    fn set_j_segments(&mut self, value: Vec<Gene>) -> Result<()> {
        self.inner.seg_js = value;
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
