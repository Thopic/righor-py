pub mod vdj;
pub mod vj;
use pyo3::prelude::*;
pub use righor::{AlignmentParameters, InferenceParameters};

#[pyfunction]
pub fn set_number_threads(nb_threads: usize) {
    righor::shared::utils::fix_number_threads(nb_threads);
}

#[pymodule]
#[pyo3(name = "_righor")]
fn righor_py(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let vdj_submod = PyModule::new(py, "vdj")?;
    let vj_submod = PyModule::new(py, "vj")?;
    vdj_submod.add_class::<vdj::PyModel>()?;
    vj_submod.add_class::<vj::PyModel>()?;
    m.add_class::<righor::vdj::GenerationResult>()?;
    m.add_class::<righor::vdj::Sequence>()?;
    m.add_class::<righor::Gene>()?;
    m.add_class::<righor::Dna>()?;
    m.add_class::<righor::AminoAcid>()?;
    m.add_class::<InferenceParameters>()?;
    m.add_class::<AlignmentParameters>()?;
    m.add_function(wrap_pyfunction!(set_number_threads, m)?)
        .unwrap();
    m.add_submodule(vdj_submod)?;
    m.add_submodule(vj_submod)?;

    Ok(())
}
