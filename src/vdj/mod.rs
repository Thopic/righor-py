//! py-binding for the VDJ model

use crate::AlignmentParameters;
use anyhow::{anyhow, Result};
use ndarray::{s, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use righor::vdj::{
    display_j_alignment, display_v_alignment, Generator, Model, ResultInference, Sequence,
};
use righor::{Dna, Gene, Modelable};
use std::fs;
use std::path::Path;

#[pyclass(name = "Model")]
#[derive(Debug, Clone)]
pub struct PyModel {
    inner: righor::vdj::Model,
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
            inner: righor::vdj::Model::load_json(&path)?,
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

    /// Return an uniform (blank slate) model with the same v/d/j genes
    /// as the current model.
    pub fn uniform(&self) -> PyResult<PyModel> {
        Ok(PyModel {
            inner: self.inner.uniform()?,
        })
    }

    /// Align one nucleotide sequence and return a `Sequence` object
    pub fn align_sequence(
        &self,
        dna_seq: &str,
        align_params: &AlignmentParameters,
    ) -> Result<Sequence> {
        let dna = righor::Dna::from_string(dna_seq)?;
        let alignment = self.inner.align_sequence(dna, align_params)?;
        Ok(alignment)
    }

    /// Given a cdr3 sequence + V/J genes return a `Sequence` object
    pub fn align_cdr3(
        &self,
        cdr3_seq: Dna,
        vgenes: Vec<Gene>,
        jgenes: Vec<Gene>,
    ) -> Result<Sequence> {
        self.inner.align_from_cdr3(cdr3_seq, vgenes, jgenes)
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
                let dna = righor::Dna::from_string(seq)?;
                let alignment = self.inner.align_sequence(dna, align_params)?;
                Ok(alignment)
            })
            .collect()
    }

    /// Evaluate the sequence and return the most likely recombination scenario
    /// as well as its probability of being generated.
    pub fn evaluate(
        &self,
        sequence: &Sequence,
        inference_params: &righor::InferenceParameters,
    ) -> Result<ResultInference> {
        self.inner.evaluate(&sequence, inference_params)
    }

    /// Run one round of expectation-maximization on the current model and return the next model.
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
        let [_, sd, sj] = *self.inner.p_vdj.shape() else {
            return Err(anyhow!("Something is wrong with the v segments"));
        };
        let mut new_p_vdj = Array3::<f64>::zeros([value.len(), sd, sj]);

        let [sdelv, _] = *self.inner.p_del_v_given_v.shape() else {
            return Err(anyhow!("Something is wrong with the v segments"));
        };
        let mut new_p_del_v_given_v = Array2::<f64>::zeros([sdelv, value.len()]);

        for (iv, v) in value.iter().enumerate() {
            match self
                .inner
                .seg_vs
                .iter()
                .enumerate()
                .find(|(_index, g)| g.name == v.name)
            {
                Some((index, _gene)) => {
                    new_p_vdj
                        .slice_mut(s![iv, .., ..])
                        .assign(&self.inner.p_vdj.slice_mut(s![index, .., ..]));
                    new_p_del_v_given_v
                        .slice_mut(s![.., iv])
                        .assign(&self.inner.p_del_v_given_v.slice_mut(s![.., index]));
                }
                None => {
                    new_p_vdj.slice_mut(s![iv, .., ..]).fill(0.);
                    new_p_del_v_given_v.slice_mut(s![.., iv]).fill(0.);
                }
            }
        }
        self.inner.seg_vs = value;
        self.inner.p_vdj = new_p_vdj;
        self.inner.p_del_v_given_v = new_p_del_v_given_v;
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
        let [sv, sd, _] = *self.inner.p_vdj.shape() else {
            return Err(anyhow!("Something is wrong with the j segments"));
        };
        let mut new_p_vdj = Array3::<f64>::zeros([sv, sd, value.len()]);

        let [sdelj, _] = *self.inner.p_del_j_given_j.shape() else {
            return Err(anyhow!("Something is wrong with the j segments"));
        };
        let mut new_p_del_j_given_j = Array2::<f64>::zeros([sdelj, value.len()]);

        for (ij, j) in value.iter().enumerate() {
            match self
                .inner
                .seg_js
                .iter()
                .enumerate()
                .find(|(_index, g)| g.name == j.name)
            {
                Some((index, _gene)) => {
                    new_p_vdj
                        .slice_mut(s![.., .., ij])
                        .assign(&self.inner.p_vdj.slice_mut(s![.., .., index]));
                    new_p_del_j_given_j
                        .slice_mut(s![.., ij])
                        .assign(&self.inner.p_del_j_given_j.slice_mut(s![.., index]));
                }
                None => {
                    new_p_vdj.slice_mut(s![.., .., ij]).fill(0.);
                    new_p_del_j_given_j.slice_mut(s![.., ij]).fill(0.);
                }
            }
        }
        self.inner.seg_js = value;
        self.inner.p_vdj = new_p_vdj;
        self.inner.p_del_j_given_j = new_p_del_j_given_j;
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

    #[setter]
    fn set_range_del_d3(&mut self, value: (i64, i64)) -> PyResult<()> {
        self.inner.range_del_d3 = value;
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_range_del_d3(&self) -> (i64, i64) {
        self.inner.range_del_d3
    }

    #[setter]
    fn set_range_del_d5(&mut self, value: (i64, i64)) -> PyResult<()> {
        self.inner.range_del_d5 = value;
        self.inner.initialize()?;
        Ok(())
    }

    #[getter]
    fn get_range_del_d5(&self) -> (i64, i64) {
        self.inner.range_del_d5
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
