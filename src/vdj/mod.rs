//! py-binding for the VDJ model

use crate::AlignmentParameters;
use anyhow::{anyhow, Result};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use righor::vdj::{
    display_j_alignment, display_v_alignment, Generator, Model, ResultInference, Sequence,
};
use righor::Gene;
use std::path::Path;

#[pyclass(name = "Model")]
#[derive(Debug, Clone)]
pub struct PyModel {
    inner: righor::vdj::Model,
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
        align_params: &AlignmentParameters,
    ) -> Result<Sequence> {
        let dna = righor::Dna::from_string(dna_seq)?;
        let alignment = self.inner.align_sequence(dna, align_params)?;
        Ok(alignment)
    }

    pub fn align_all_sequences(
        &self,
        dna_seqs: Vec<String>,
        align_params: &AlignmentParameters,
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
        self.inner.evaluate(&sequence, inference_params)
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
            &self.inner,
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
            &self.inner,
            align_params,
        ))
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
    fn get_error_rate(&self) -> f64 {
        self.inner.error_rate
    }

    #[setter]
    fn set_error_rate(&mut self, value: f64) -> Result<()> {
        self.inner.error_rate = value;
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_d_segments(&self) -> Vec<Gene> {
        self.inner.seg_ds.to_owned()
    }

    #[setter]
    fn set_d_segments(&mut self, value: Vec<Gene>) -> Result<()> {
        self.inner.seg_ds = value;
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

    #[getter]
    fn get_p_v(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.p_v.to_owned().into_pyarray(py).to_owned()
    }

    /// Return the marginal on (D, J)
    #[getter]
    fn get_p_dj(&self, py: Python) -> Py<PyArray2<f64>> {
        self.inner.p_dj.to_owned().into_pyarray(py).to_owned()
    }

    #[getter(p_vdj)]
    fn get_p_vdj(&self, py: Python) -> Py<PyArray3<f64>> {
        self.inner.p_vdj.to_owned().into_pyarray(py).to_owned()
    }

    #[setter(p_vdj)]
    fn set_p_vdj(&mut self, py: Python, value: Py<PyArray3<f64>>) -> PyResult<()> {
        self.inner.p_vdj = value.as_ref(py).to_owned_array();
        self.inner.set_p_vdj(&self.inner.p_vdj.clone())?;
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_p_ins_vd(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.p_ins_vd.to_owned().into_pyarray(py).to_owned()
    }

    #[setter]
    fn set_p_ins_vd(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.inner.p_ins_vd = value.as_ref(py).to_owned_array();
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_p_ins_dj(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.p_ins_dj.to_owned().into_pyarray(py).to_owned()
    }

    #[setter]
    fn set_p_ins_dj(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.inner.p_ins_dj = value.as_ref(py).to_owned_array();
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
    fn get_p_del_d5_del_d3(&self, py: Python) -> Py<PyArray3<f64>> {
        self.inner
            .p_del_d5_del_d3
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }

    #[setter]
    fn set_p_del_d5_del_d3(&mut self, py: Python, value: Py<PyArray3<f64>>) -> PyResult<()> {
        self.inner.p_del_d5_del_d3 = value.as_ref(py).to_owned_array();
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_markov_coefficients_vd(&self, py: Python) -> Py<PyArray2<f64>> {
        self.inner
            .markov_coefficients_vd
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }

    #[setter]
    fn set_markov_coefficients_vd(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.inner.markov_coefficients_vd = value.as_ref(py).to_owned_array();
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_markov_coefficients_dj(&self, py: Python) -> Py<PyArray2<f64>> {
        self.inner
            .markov_coefficients_dj
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }

    #[setter]
    fn set_markov_coefficients_dj(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.inner.markov_coefficients_dj = value.as_ref(py).to_owned_array();
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_first_nt_bias_ins_vd(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner
            .first_nt_bias_ins_vd
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }

    #[getter]
    fn get_first_nt_bias_ins_dj(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner
            .first_nt_bias_ins_dj
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }

    fn extract_features(&self, res: &ResultInference) -> Result<PyModel> {
        let feat = res
            .features
            .clone()
            .ok_or(anyhow!("No feature data in this inference result."))?;
        let model = self.inner.from_features(&feat)?;
        Ok(PyModel { inner: model })
    }
}

#[pyfunction]
pub fn test() -> Result<PyModel> {
    let igor_model = Model::load_from_files(
        Path::new("models/human/t_beta/tmp1/model_params.txt"),
        Path::new("models/human/t_beta/tmp1/model_marginals.txt"),
        Path::new("models/human/t_beta/V_gene_CDR3_anchors.csv"),
        Path::new("models/human/t_beta/J_gene_CDR3_anchors.csv"),
    )?;

    let mut uniform_model = igor_model.uniform()?;

    let align_params = AlignmentParameters::default();
    let inference_params = righor::InferenceParameters::default();
    let mut generator = Generator::new(igor_model.clone(), Some(42), None, None)?;

    let mut seq = Vec::new();
    for _ in 0..10 {
        let s = righor::Dna::from_string(&generator.generate(false).full_seq)?;
        println!("{}", s.get_string());
        let als = uniform_model.align_sequence(s.clone(), &align_params)?;
        if !(als.v_genes.is_empty() || als.j_genes.is_empty()) {
            seq.push(als);
        }
    }
    for ii in 0..1 {
        uniform_model.infer(&seq, &inference_params)?;
        println!("{:?}", ii);
    }
    Ok(PyModel {
        inner: uniform_model,
    })
}

#[pyfunction]
pub fn test_identity(m1: PyModel, m2: PyModel) {
    println!("{}", (m1.inner.seg_vs == m2.inner.seg_vs));
    println!("{}", (m1.inner.seg_js == m2.inner.seg_js));
    println!("{}", (m1.inner.seg_ds == m2.inner.seg_ds));
    println!(
        "{}",
        (m1.inner.seg_vs_sanitized == m2.inner.seg_vs_sanitized)
    );
    println!(
        "{}",
        (m1.inner.seg_js_sanitized == m2.inner.seg_js_sanitized)
    );
    println!(
        "{}",
        (m1.inner
            .p_d_given_vj
            .relative_eq(&m2.inner.p_d_given_vj, 1e-4, 1e-4))
    );
    println!("{} {}", m1.inner.p_j_given_v, m2.inner.p_j_given_v);
    println!("{}", (m1.inner.p_v.relative_eq(&m2.inner.p_v, 1e-4, 1e-4)));
    println!("{} {}", m1.inner.p_ins_vd, m2.inner.p_ins_vd);
    println!(
        "{}",
        (m1.inner
            .p_ins_dj
            .relative_eq(&m2.inner.p_ins_dj, 1e-4, 1e-4))
    );
    println!(
        "{}",
        (m1.inner
            .p_del_v_given_v
            .relative_eq(&m2.inner.p_del_v_given_v, 1e-4, 1e-4))
    );
    println!(
        "{}",
        (m1.inner
            .p_del_j_given_j
            .relative_eq(&m2.inner.p_del_j_given_j, 1e-4, 1e-4))
    );
    println!(
        "{}",
        (m1.inner
            .p_del_d5_del_d3
            .relative_eq(&m2.inner.p_del_d5_del_d3, 1e-4, 1e-4))
    );
    println!(
        "{}",
        (m1.inner
            .markov_coefficients_vd
            .relative_eq(&m2.inner.markov_coefficients_vd, 1e-4, 1e-4,))
    );
    println!(
        "{}",
        (m1.inner
            .markov_coefficients_dj
            .relative_eq(&m2.inner.markov_coefficients_dj, 1e-4, 1e-4,))
    );
    println!("{}", (m1.inner.range_del_v == m2.inner.range_del_v));
    println!("{}", (m1.inner.range_del_j == m2.inner.range_del_j));
    println!("{}", (m1.inner.range_del_d3 == m2.inner.range_del_d3));
    println!("{}", (m1.inner.range_del_d5 == m2.inner.range_del_d5));
    println!(
        "{}",
        (m1.inner.error_rate - m2.inner.error_rate).abs() < 1e-4
    );
    println!("{}", (m1.inner.thymic_q == m2.inner.thymic_q));
    println!(
        "{}",
        (m1.inner.p_dj.relative_eq(&m2.inner.p_dj, 1e-4, 1e-4))
    );
    println!(
        "{}",
        (m1.inner.p_vdj.relative_eq(&m2.inner.p_vdj, 1e-4, 1e-4))
    );
}
