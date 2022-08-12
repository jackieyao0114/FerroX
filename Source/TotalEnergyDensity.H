#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void CalculateTDGL_RHS(
                // MultiFab&                GL_rhs,
                Array<MultiFab, AMREX_SPACEDIM>& GL_rhs,
                // MultiFab&                       P_old,
                Array<MultiFab, AMREX_SPACEDIM>& P_old,
                MultiFab&                       PoissonPhi,
                MultiFab&                       Gamma,
                Real                            FE_lo,
                Real                            FE_hi,
                Real                            DE_lo,
                Real                            DE_hi,
                Real                            SC_lo,
                Real                            SC_hi,
                int                             P_BC_flag_lo,
                int                             P_BC_flag_hi,
                Real                            Phi_Bc_lo,
                Real                            Phi_Bc_hi,
                Real                            alpha,
                Real                            beta,
                Real                            gamma,
                Real                            g11,
                Real                            g44,
                Real                            lambda,
                amrex::GpuArray<amrex::Real, 3> prob_lo,
                amrex::GpuArray<amrex::Real, 3> prob_hi,
                const Geometry& geom);

/**
 * Perform derivative along x on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDx (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, const Geometry& geom) {

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    amrex::Real const inv_dx = 1./dx[0];
    return inv_dx*( F(i+1,j,k) - F(i,j,k) );
}

/**
 * Perform derivative along x on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDx (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, const Geometry& geom) {

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    amrex::Real const inv_dx = 1./dx[0];
    return inv_dx*( F(i,j,k) - F(i-1,j,k) );
}

/**
 * Perform derivative along x on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real CenteredDPDx (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, const Geometry& geom) {

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    amrex::Real const inv_dx = 1./(2*dx[0]);
    return inv_dx*( F(i+1,j,k) - F(i-1,j,k) );
}

/**
 * Perform derivative along y on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDy (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, const Geometry& geom) {

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    amrex::Real const inv_dy = 1./dx[1];
    return inv_dy*( F(i,j+1,k) - F(i,j,k) );
}

/**
 * Perform derivative along y on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real CenteredDPDy (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, const Geometry& geom) {

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    amrex::Real const inv_dy = 1./(2*dx[1]);
    return inv_dy*( F(i,j+1,k) - F(i,j-1,k) );
}

/**
 * Perform derivative along y on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDy (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, const Geometry& geom) {

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    amrex::Real const inv_dy = 1./dx[1];
    return inv_dy*( F(i,j,k) - F(i,j-1,k) );
}

/**
 * Perform derivative along z on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDz (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, const Geometry& geom) {

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    amrex::Real const inv_dz = 1./dx[2];
    return inv_dz*( F(i,j,k+1) - F(i,j,k) );
}

/**
 * Perform derivative along z on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real CenteredDz (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, const Geometry& geom) {

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    amrex::Real const inv_dz = 1./(2.*dx[2]);
    return inv_dz*( F(i,j,k+1) - F(i,j,k-1) );
}

/**
 * Perform derivative along z on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDz (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, const Geometry& geom) {

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    amrex::Real const inv_dz = 1./dx[2];
    return inv_dz*( F(i,j,k) - F(i,j,k-1) );
}

/**
 * Perform derivative along z on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DPDz2 (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, const Geometry& geom) {

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    amrex::Real const inv_dz = 1./dx[2];
    return inv_dz*( F(i,j,k) - F(i,j,k-1) );
}