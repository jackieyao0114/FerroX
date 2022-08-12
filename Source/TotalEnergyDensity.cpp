#include "TotalEnergyDensity.H"

void CalculateTDGL_RHS(
    // MultiFab&                GL_rhs,
    Array<MultiFab, AMREX_SPACEDIM> &GL_rhs,
    // MultiFab&                       P_old,
    Array<MultiFab, AMREX_SPACEDIM> &P_old,
    MultiFab &PoissonPhi,
    MultiFab &Gamma,
    Real FE_lo,
    Real FE_hi,
    Real DE_lo,
    Real DE_hi,
    Real SC_lo,
    Real SC_hi,
    int P_BC_flag_lo,
    int P_BC_flag_hi,
    Real Phi_Bc_lo,
    Real Phi_Bc_hi,
    Real alpha,
    Real beta,
    Real gamma,
    Real g11,
    Real g44,
    Real lambda,
    Real alpha_12,
    Real alpha_112,
    Real alpha_123,
    amrex::GpuArray<amrex::Real, 3> prob_lo,
    amrex::GpuArray<amrex::Real, 3> prob_hi,
    const Geometry &geom)
{
  // loop over boxes
  for (MFIter mfi(P_old[0]); mfi.isValid(); ++mfi)
  {
    const Box &bx = mfi.validbox();

    // extract dx from the geometry object
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    const Array4<Real> &GL_RHS_x = GL_rhs[0].array(mfi);
    const Array4<Real> &GL_RHS_y = GL_rhs[1].array(mfi);
    const Array4<Real> &GL_RHS_z = GL_rhs[2].array(mfi);
    const Array4<Real> &pOld_x = P_old[0].array(mfi);
    const Array4<Real> &pOld_y = P_old[1].array(mfi);
    const Array4<Real> &pOld_z = P_old[2].array(mfi);
    const Array4<Real> &phi = PoissonPhi.array(mfi);
    const Array4<Real> &Gam = Gamma.array(mfi);

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                       {
                Real grad_term, phi_term, d2P_z;
                Real z    = prob_lo[2] + (k+0.5) * dx[2];
                Real z_hi = prob_lo[2] + (k+1.5) * dx[2];
                Real z_lo = prob_lo[2] + (k-0.5) * dx[2];

                if(z_lo < prob_lo[2]){ //Bottom metal

                  grad_term = 0.0;
                  phi_term = (-4.*Phi_Bc_lo + 3.*phi(i,j,k) + phi(i,j,k+1))/(3.*dx[2]);
                  //phi_term = (phi(i,j,k+1) - phi(i,j,k)) / (dx[2]);

                } else if(z < DE_hi){ //Below FE-DE interface

                  grad_term = 0.0;
                  phi_term = (phi(i,j,k+1) - phi(i,j,k-1)) / (2.*dx[2]);

                } else if (DE_hi > z_lo && DE_hi <= z) { //FE side of FE-DE interface

                  if(P_BC_flag_lo == 0){
                    Real P_int = 0.0;
                    d2P_z = 4.*(2.*P_int - 3.*pOld(i,j,k) + pOld(i,j,k+1))/3./dx[2]/dx[2];//2nd Order
                    grad*centeredDz2
                    

                  } else if (P_BC_flag_lo == 1){
                    Real P_int = pOld(i,j,k)/(1 + dx[2]/2/lambda);
                    Real dPdz = P_int/lambda;
                    d2P_z = (-dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/dx[2]/dx[2];//2nd Order
                  } else if (P_BC_flag_lo == 2){
                    Real dPdz = 0.;
                    d2P_z = (-dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/dx[2]/dx[2];//2nd Order
                  }

                  grad_term = g11 * d2P_z;
                  phi_term = (phi(i,j,k+1) - phi(i,j,k-1)) / (2.*dx[2]);

                } else if (z_hi > prob_hi[2]){ //Top metal

                        if(P_BC_flag_hi == 0){
                    Real P_int = 0.0;
                    d2P_z = 4.*(2.*P_int - 3.*pOld(i,j,k) + pOld(i,j,k-1))/3./dx[2]/dx[2];//2nd Order
                  } else if (P_BC_flag_hi == 1){
                    Real P_int = pOld(i,j,k)/(1 - dx[2]/2/lambda);
                    Real dPdz = P_int/lambda;
                    d2P_z = (dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k-1))/dx[2]/dx[2];//2nd Order
                  } else if (P_BC_flag_hi == 2){
                    Real dPdz = 0.;
                    d2P_z = (dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k-1))/dx[2]/dx[2];//2nd Order
                  }

                  grad_term = g11 * d2P_z;
                  phi_term = (4.*Phi_Bc_hi - 3.*phi(i,j,k) - phi(i,j,k-1))/(3.*dx[2]);

                } else{ //inside FE

                  grad_term = g11 * (pOld(i,j,k+1) - 2.*pOld(i,j,k) + pOld(i,j,k-1)) / (dx[2]*dx[2]);
                  phi_term = (phi(i,j,k+1) - phi(i,j,k-1)) / (2.*dx[2]);

                  dPx_dz_hi = CenteredDz(pOld_z, i+1, j, k, geom);
                  dPx_dz_lo = CenteredDz(pOld_z, i-1, j, k, geom);

                  dFdPz_grad = - g11 * (UpwardDz (pOld_z, i, j, k, geom) - DownwardDz(pOld_z, i, j, k, geom)) / dx[2]
                               - g44 * (UpwardDx (pOld_z, i, j, k, geom) - DownwardDx(pOld_z, i, j, k, geom)) / dx[0]
                               - g44 * (UpwardDy (pOld_z, i, j, k, geom) - DownwardDy(pOld_z, i, j, k, geom)) / dx[1]
                               - g12 * (Centered) / (2.* dx[0])


                }

                // GL_RHS(i,j,k)  = -1.0 * Gam(i,j,k) *
                //     (  alpha*pOld(i,j,k) + beta*std::pow(pOld(i,j,k),3.) + gamma*std::pow(pOld(i,j,k),5.)
                //      - g44 * (pOld(i+1,j,k) - 2.*pOld(i,j,k) + pOld(i-1,j,k)) / (dx[0]*dx[0])
                //      - g44 * (pOld(i,j+1,k) - 2.*pOld(i,j,k) + pOld(i,j-1,k)) / (dx[1]*dx[1])
                //      - grad_term
                //      + phi_term
                //     );

                Real dFdPx_Landau = alpha*pOld_x(i,j,k) + beta*std::pow(pOld_x(i,j,k),3.) + gamma*std::pow(pOld_x(i,j,k),5.)
                                    + 2 * alpha_12 * pOld_x(i,j,k) * std::pow(pOld_y(i,j,k),2.)
                                    + 2 * alpha_12 * pOld_x(i,j,k) * std::pow(pOld_z(i,j,k),2.)
                                    + 4 * alpha_112 * std::pow(pOld_x(i,j,k),3.) * (std::pow(pOld_y(i,j,k),2.) + std::pow(pOld_z(i,j,k),2.))
                                    + 2 * alpha_112 * pOld_x(i,j,k) * std::pow(pOld_y(i,j,k),4.)
                                    + 2 * alpha_112 * pOld_x(i,j,k) * std::pow(pOld_z(i,j,k),4.)
                                    + 2 * alpha_123 * pOld_x(i,j,k) * std::pow(pOld_y(i,j,k),2.) * std::pow(pOld_z(i,j,k),2.);

                Real dFdPy_Landau = alpha*pOld_y(i,j,k) + beta*std::pow(pOld_y(i,j,k),3.) + gamma*std::pow(pOld_y(i,j,k),5.)
                                    + 2 * alpha_12 * pOld_y(i,j,k) * std::pow(pOld_x(i,j,k),2.)
                                    + 2 * alpha_12 * pOld_y(i,j,k) * std::pow(pOld_z(i,j,k),2.)
                                    + 4 * alpha_112 * std::pow(pOld_y(i,j,k),3.) * (std::pow(pOld_x(i,j,k),2.) + std::pow(pOld_z(i,j,k),2.))
                                    + 2 * alpha_112 * pOld_y(i,j,k) * std::pow(pOld_x(i,j,k),4.)
                                    + 2 * alpha_112 * pOld_y(i,j,k) * std::pow(pOld_z(i,j,k),4.)
                                    + 2 * alpha_123 * pOld_y(i,j,k) * std::pow(pOld_x(i,j,k),2.) * std::pow(pOld_z(i,j,k),2.);
                
                Real dFdPz_Landau = alpha*pOld_z(i,j,k) + beta*std::pow(pOld_z(i,j,k),3.) + gamma*std::pow(pOld_z(i,j,k),5.)
                                    + 2 * alpha_12 * pOld_z(i,j,k) * std::pow(pOld_x(i,j,k),2.)
                                    + 2 * alpha_12 * pOld_z(i,j,k) * std::pow(pOld_y(i,j,k),2.)
                                    + 4 * alpha_112 * std::pow(pOld_z(i,j,k),3.) * (std::pow(pOld_x(i,j,k),2.) + std::pow(pOld_y(i,j,k),2.))
                                    + 2 * alpha_112 * pOld_z(i,j,k) * std::pow(pOld_x(i,j,k),4.)
                                    + 2 * alpha_112 * pOld_z(i,j,k) * std::pow(pOld_y(i,j,k),4.)
                                    + 2 * alpha_123 * pOld_z(i,j,k) * std::pow(pOld_x(i,j,k),2.) * std::pow(pOld_y(i,j,k),2.);

                GL_RHS_x(i,j,k)  = -1.0 * Gam(i,j,k) *
                    (  dFdPx_Landau
                     - g44 * (pOld_x(i+1,j,k) - 2.*pOld_x(i,j,k) + pOld_x(i-1,j,k)) / (dx[0]*dx[0])
                     - g44 * (pOld_x(i,j+1,k) - 2.*pOld_x(i,j,k) + pOld_x(i,j-1,k)) / (dx[1]*dx[1])
                     - grad_term
                     + phi_term
                    );

                GL_RHS_y(i,j,k)  = -1.0 * Gam(i,j,k) *
                    (  dFdPy_Landau
                     - g44 * (pOld_y(i+1,j,k) - 2.*pOld_y(i,j,k) + pOld_y(i-1,j,k)) / (dx[0]*dx[0])
                     - g44 * (pOld_y(i,j+1,k) - 2.*pOld_y(i,j,k) + pOld_y(i,j-1,k)) / (dx[1]*dx[1])
                     - grad_term
                     + phi_term
                    );

                GL_RHS_z(i,j,k)  = -1.0 * Gam(i,j,k) *
                    (  dFdPz_Landau
                     - g44 * (pOld_z(i+1,j,k) - 2.*pOld_z(i,j,k) + pOld_z(i-1,j,k)) / (dx[0]*dx[0])
                     - g44 * (pOld_z(i,j+1,k) - 2.*pOld_z(i,j,k) + pOld_z(i,j-1,k)) / (dx[1]*dx[1])
                     - grad_term
                     + phi_term
                    ); });
  }
}
