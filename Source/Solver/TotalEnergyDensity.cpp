#include "TotalEnergyDensity.H"
#include "DerivativeAlgorithm.H"
#include "AMReX_CONSTANTS.H"


void CalculateTDGL_RHS(Array<MultiFab, AMREX_SPACEDIM> &GL_rhs,
                Array<MultiFab, AMREX_SPACEDIM> &P_old,
                Array<MultiFab, AMREX_SPACEDIM> &E,
                MultiFab&                       BigGamma,
                MultiFab&                       alpha,
                MultiFab&                       beta,
                MultiFab&                       gamma,
                MultiFab&                       g11,
                MultiFab&                       g44,
                MultiFab&                       g44_p,
                MultiFab&                       g12,
                MultiFab&                       alpha_12,
                MultiFab&                       alpha_112,
                MultiFab&                       alpha_123,
                MultiFab&                 MaterialMask,
                MultiFab&                 tphaseMask,
                MultiFab& angle_alpha, 
                MultiFab& angle_beta, 
                MultiFab& angle_theta,
                const Geometry& geom,
		        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi)
{
//        Real average_P_r = 0.;
//        Real total_P_r = 0.;
//        Real FE_index_counter = 0.;
//
//        Compute_P_av(P_old, total_P_r, MaterialMask, FE_index_counter, average_P_r);
//
//        //Calculate integrated electrode charge (Qe) based on eq 13 of https://pubs.aip.org/aip/jap/article/44/8/3379/6486/Depolarization-fields-in-thin-ferroelectric-films
//        Real FE_thickness = FE_hi[2] - FE_lo[2];
//	Real one_m_theta = 1.0 - theta_dep;
//        Real E_dep = average_P_r/(epsilon_0*epsilonZ_fe)*one_m_theta;
//
	// loop over boxes
        for ( MFIter mfi(P_old[0]); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real> &GL_RHS_p = GL_rhs[0].array(mfi);
            const Array4<Real> &GL_RHS_q = GL_rhs[1].array(mfi);
            const Array4<Real> &GL_RHS_r = GL_rhs[2].array(mfi);
            const Array4<Real> &pOld_p = P_old[0].array(mfi);
            const Array4<Real> &pOld_q = P_old[1].array(mfi);
            const Array4<Real> &pOld_r = P_old[2].array(mfi);
            const Array4<Real> &Ep = E[0].array(mfi);
            const Array4<Real> &Eq = E[1].array(mfi);
            const Array4<Real> &Er = E[2].array(mfi);
            const Array4<Real>& mat_BigGamma = BigGamma.array(mfi);
            const Array4<Real>& mat_alpha = alpha.array(mfi);
            const Array4<Real>& mat_beta = beta.array(mfi);
            const Array4<Real>& mat_gamma = gamma.array(mfi);
            const Array4<Real>& mat_g11 = g11.array(mfi);
            const Array4<Real>& mat_g44 = g44.array(mfi);
            const Array4<Real>& mat_g44_p = g44_p.array(mfi);
            const Array4<Real>& mat_g12 = g12.array(mfi);
            const Array4<Real>& mat_alpha_12 = alpha_12.array(mfi);
            const Array4<Real>& mat_alpha_112 = alpha_112.array(mfi);
            const Array4<Real>& mat_alpha_123 = alpha_123.array(mfi);
            const Array4<Real>& mask = MaterialMask.array(mfi);
            const Array4<Real>& tphase = tphaseMask.array(mfi);

            const Array4<Real> &angle_alpha_arr = angle_alpha.array(mfi);
            const Array4<Real> &angle_beta_arr = angle_beta.array(mfi);
            const Array4<Real> &angle_theta_arr = angle_theta.array(mfi);


            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {

               //Convert Euler angles from degrees to radians
               amrex::Real Pi = 3.14159265358979323846; 
               amrex::Real alpha_rad = Pi/180.*angle_alpha_arr(i,j,k);
               amrex::Real beta_rad =  Pi/180.*angle_beta_arr(i,j,k);
               amrex::Real theta_rad = Pi/180.*angle_theta_arr(i,j,k);
  
               amrex::Real R_11, R_12, R_13, R_21, R_22, R_23, R_31, R_32, R_33;
  
               if(use_Euler_angles){
                  R_11 = cos(alpha_rad)*cos(theta_rad) - cos(beta_rad)*sin(alpha_rad)*sin(theta_rad);  
                  R_12 = sin(alpha_rad)*cos(theta_rad) + cos(beta_rad)*cos(alpha_rad)*sin(theta_rad);  
                  R_13 = sin(beta_rad)*sin(theta_rad);  
                  R_21 = -cos(beta_rad)*cos(theta_rad)*sin(alpha_rad) - cos(alpha_rad)*sin(theta_rad);  
                  R_22 = cos(beta_rad)*cos(alpha_rad)*cos(theta_rad) - sin(alpha_rad)*sin(theta_rad);  
                  R_23 = sin(beta_rad)*cos(theta_rad);  
                  R_31 = sin(alpha_rad)*sin(beta_rad);  
                  R_32 = -cos(alpha_rad)*sin(beta_rad);  
                  R_33 = cos(beta_rad);  
               } else {
                  R_11 = cos(beta_rad)*cos(theta_rad);  
                  R_12 = sin(alpha_rad)*sin(beta_rad)*cos(theta_rad) - cos(alpha_rad)*sin(theta_rad);  
                  R_13 = cos(alpha_rad)*sin(beta_rad)*cos(theta_rad) + sin(alpha_rad)*sin(theta_rad);  
                  R_21 = cos(beta_rad)*sin(theta_rad);  
                  R_22 = sin(beta_rad)*sin(alpha_rad)*sin(theta_rad) + cos(alpha_rad)*cos(theta_rad);
                  R_23 = cos(alpha_rad)*sin(beta_rad)*sin(theta_rad) - sin(alpha_rad)*cos(theta_rad);
                  R_31 = -sin(beta_rad);
                  R_32 = sin(alpha_rad)*cos(beta_rad);
                  R_33 = cos(alpha_rad)*cos(beta_rad);
               }

                Real dFdPp_Landau = mat_alpha(i,j,k)*pOld_p(i,j,k) + mat_beta(i,j,k)*std::pow(pOld_p(i,j,k),3.) + mat_gamma(i,j,k)*std::pow(pOld_p(i,j,k),5.)
                                    + 2. * mat_alpha_12(i,j,k) * pOld_p(i,j,k) * std::pow(pOld_q(i,j,k),2.)
                                    + 2. * mat_alpha_12(i,j,k) * pOld_p(i,j,k) * std::pow(pOld_r(i,j,k),2.)
                                    + 4. * mat_alpha_112(i,j,k) * std::pow(pOld_p(i,j,k),3.) * (std::pow(pOld_q(i,j,k),2.) + std::pow(pOld_r(i,j,k),2.))
                                    + 2. * mat_alpha_112(i,j,k) * pOld_p(i,j,k) * std::pow(pOld_q(i,j,k),4.)
                                    + 2. * mat_alpha_112(i,j,k) * pOld_p(i,j,k) * std::pow(pOld_r(i,j,k),4.)
                                    + 2. * mat_alpha_123(i,j,k) * pOld_p(i,j,k) * std::pow(pOld_q(i,j,k),2.) * std::pow(pOld_r(i,j,k),2.);

                Real dFdPq_Landau = mat_alpha(i,j,k)*pOld_q(i,j,k) + mat_beta(i,j,k)*std::pow(pOld_q(i,j,k),3.) + mat_gamma(i,j,k)*std::pow(pOld_q(i,j,k),5.)
                                    + 2. * mat_alpha_12(i,j,k) * pOld_q(i,j,k) * std::pow(pOld_p(i,j,k),2.)
                                    + 2. * mat_alpha_12(i,j,k) * pOld_q(i,j,k) * std::pow(pOld_r(i,j,k),2.)
                                    + 4. * mat_alpha_112(i,j,k) * std::pow(pOld_q(i,j,k),3.) * (std::pow(pOld_p(i,j,k),2.) + std::pow(pOld_r(i,j,k),2.))
                                    + 2. * mat_alpha_112(i,j,k) * pOld_q(i,j,k) * std::pow(pOld_p(i,j,k),4.)
                                    + 2. * mat_alpha_112(i,j,k) * pOld_q(i,j,k) * std::pow(pOld_r(i,j,k),4.)
                                    + 2. * mat_alpha_123(i,j,k) * pOld_q(i,j,k) * std::pow(pOld_p(i,j,k),2.) * std::pow(pOld_r(i,j,k),2.);
                
                Real dFdPr_Landau = mat_alpha(i,j,k)*pOld_r(i,j,k) + mat_beta(i,j,k)*std::pow(pOld_r(i,j,k),3.) + mat_gamma(i,j,k)*std::pow(pOld_r(i,j,k),5.)
                                    + 2. * mat_alpha_12(i,j,k) * pOld_r(i,j,k) * std::pow(pOld_p(i,j,k),2.)
                                    + 2. * mat_alpha_12(i,j,k) * pOld_r(i,j,k) * std::pow(pOld_q(i,j,k),2.)
                                    + 4. * mat_alpha_112(i,j,k) * std::pow(pOld_r(i,j,k),3.) * (std::pow(pOld_p(i,j,k),2.) + std::pow(pOld_q(i,j,k),2.))
                                    + 2. * mat_alpha_112(i,j,k) * pOld_r(i,j,k) * std::pow(pOld_p(i,j,k),4.)
                                    + 2. * mat_alpha_112(i,j,k) * pOld_r(i,j,k) * std::pow(pOld_q(i,j,k),4.)
                                    + 2. * mat_alpha_123(i,j,k) * pOld_r(i,j,k) * std::pow(pOld_p(i,j,k),2.) * std::pow(pOld_q(i,j,k),2.);

                Real dFdPp_grad = - mat_g11(i,j,k) * DoubleDPDx(pOld_p, mask, i, j, k, dx)
                                  - (mat_g44(i,j,k) + mat_g44_p(i,j,k)) * DoubleDPDy(pOld_p, mask, i, j, k, dx)
                                  - (mat_g44(i,j,k) + mat_g44_p(i,j,k)) * DoubleDPDz(pOld_p, mask, i, j, k, dx)
                                  - (mat_g12(i,j,k) + mat_g44(i,j,k) - mat_g44_p(i,j,k)) * DoubleDPDxDy(pOld_q, mask, i, j, k, dx)  // d2P/dxdy
                                  - (mat_g12(i,j,k) + mat_g44(i,j,k) - mat_g44_p(i,j,k)) * DoubleDPDxDz(pOld_r, mask, i, j, k, dx); // d2P/dxdz
                
                Real dFdPq_grad = - mat_g11(i,j,k) * DoubleDPDy(pOld_q, mask, i, j, k, dx)
                                  - (mat_g44(i,j,k) - mat_g44_p(i,j,k)) * DoubleDPDx(pOld_q, mask, i, j, k, dx)
                                  - (mat_g44(i,j,k) - mat_g44_p(i,j,k)) * DoubleDPDz(pOld_q, mask, i, j, k, dx)
                                  - (mat_g12(i,j,k) + mat_g44(i,j,k) + mat_g44_p(i,j,k)) * DoubleDPDxDy(pOld_p, mask, i, j, k, dx) // d2P/dxdy
                                  - (mat_g12(i,j,k) + mat_g44(i,j,k) - mat_g44_p(i,j,k)) * DoubleDPDyDz(pOld_r, mask, i, j, k, dx);// d2P/dydz

                Real dFdPr_grad = - mat_g11(i,j,k) * ( R_31*R_31*DoubleDPDx(pOld_r, mask, i, j, k, dx)
                                           +R_32*R_32*DoubleDPDy(pOld_r, mask, i, j, k, dx)
                                           +R_33*R_33*DoubleDPDz(pOld_r, mask, i, j, k, dx)
                                           +2.*R_31*R_32*DoubleDPDxDy(pOld_r, mask, i, j, k, dx)
                                           +2.*R_32*R_33*DoubleDPDyDz(pOld_r, mask, i, j, k, dx)
                                           +2.*R_33*R_31*DoubleDPDxDz(pOld_r, mask, i, j, k, dx))
                                           
                                  - (mat_g44(i,j,k) - mat_g44_p(i,j,k)) * ( R_11*R_11*DoubleDPDx(pOld_r, mask, i, j, k, dx) 
                                                     +R_12*R_12*DoubleDPDy(pOld_r, mask, i, j, k, dx) 
                                                     +R_13*R_13*DoubleDPDz(pOld_r, mask, i, j, k, dx) 
                                                     +2.*R_11*R_12*DoubleDPDxDy(pOld_r, mask, i, j, k, dx) 
                                                     +2.*R_12*R_13*DoubleDPDyDz(pOld_r, mask, i, j, k, dx) 
                                                     +2.*R_13*R_11*DoubleDPDxDz(pOld_r, mask, i, j, k, dx))

                                  - (mat_g44(i,j,k) - mat_g44_p(i,j,k)) * ( R_21*R_21*DoubleDPDx(pOld_r, mask, i, j, k, dx)
                                                     +R_22*R_22*DoubleDPDy(pOld_r, mask, i, j, k, dx)
                                                     +R_23*R_23*DoubleDPDz(pOld_r, mask, i, j, k, dx)
                                                     +2.*R_21*R_22*DoubleDPDxDy(pOld_r, mask, i, j, k, dx)
                                                     +2.*R_22*R_23*DoubleDPDyDz(pOld_r, mask, i, j, k, dx)
                                                     +2.*R_23*R_21*DoubleDPDxDz(pOld_r, mask, i, j, k, dx))

                                  - (mat_g44(i,j,k) + mat_g44_p(i,j,k) + mat_g12(i,j,k)) * DoubleDPDyDz(pOld_q, mask, i, j, k, dx) // d2P/dydz
                                  - (mat_g44(i,j,k) + mat_g44_p(i,j,k) + mat_g12(i,j,k)) * DoubleDPDxDz(pOld_p, mask, i, j, k, dx); // d2P/dxdz

                GL_RHS_p(i,j,k) = -1.0 * mat_BigGamma(i,j,k) *
                    (  dFdPp_Landau
                     + dFdPp_grad
		             - Ep(i,j,k)
                    );

                GL_RHS_q(i,j,k) = -1.0 * mat_BigGamma(i,j,k) *
                    (  dFdPq_Landau
                     + dFdPq_grad
		             - Eq(i,j,k)
                    );

                GL_RHS_r(i,j,k) = -1.0 * mat_BigGamma(i,j,k) *
                    (  dFdPr_Landau
                     + dFdPr_grad
		             - Er(i,j,k) //+ E_dep
                    );

                if (is_polarization_scalar == 1){
		            GL_RHS_p(i,j,k) = 0.0;
		            GL_RHS_q(i,j,k) = 0.0;
		        }

		//set t_phase GL_RHS_r to zero so that it stays zero. It is initialized to zero in t-phase as well
                //if(x <= t_phase_hi[0] && x >= t_phase_lo[0] && y <= t_phase_hi[1] && y >= t_phase_lo[1] && z <= t_phase_hi[2] && z >= t_phase_lo[2]){
                if (tphase(i,j,k) == 1.0){
		            GL_RHS_p(i,j,k) = 0.0;
		            GL_RHS_q(i,j,k) = 0.0;
		            GL_RHS_r(i,j,k) = 0.0;
                }
            });
        }
}

void Compute_P_Sum(const std::array<MultiFab, AMREX_SPACEDIM>& P, Real& sum)
{

     // Initialize to zero
     sum = 0.;

     ReduceOps<ReduceOpSum> reduce_op;

     ReduceData<Real> reduce_data(reduce_op);
     using ReduceTuple = typename decltype(reduce_data)::Type;

     for (MFIter mfi(P[2],TilingIfNotGPU()); mfi.isValid(); ++mfi)
     {
         const Box& bx = mfi.tilebox();
         const Box& bx_grid = mfi.validbox();

         auto const& fab = P[2].array(mfi);

         reduce_op.eval(bx, reduce_data,
         [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
         {
             return {fab(i,j,k)};
         });
     }

     sum = amrex::get<0>(reduce_data.value());
     ParallelDescriptor::ReduceRealSum(sum);
}


void Compute_P_index_Sum(const MultiFab& MaterialMask, Real& count)
{

     // Initialize to zero
     count = 0.;

     ReduceOps<ReduceOpSum> reduce_op;

     ReduceData<Real> reduce_data(reduce_op);
     using ReduceTuple = typename decltype(reduce_data)::Type;

     for (MFIter mfi(MaterialMask, TilingIfNotGPU()); mfi.isValid(); ++mfi)
     {
         const Box& bx = mfi.tilebox();
         const Box& bx_grid = mfi.validbox();

         auto const& fab = MaterialMask.array(mfi);

         reduce_op.eval(bx, reduce_data,
         [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
         {
             if(fab(i,j,k) == 0.) {
               return {1.};
             } else {
               return {0.};
             }

         });
     }

     count = amrex::get<0>(reduce_data.value());
     ParallelDescriptor::ReduceRealSum(count);
}

void Compute_P_av(const std::array<MultiFab, AMREX_SPACEDIM>& P, Real& sum, const MultiFab& MaterialMask, Real& count, Real& P_av_z)
{
     Compute_P_Sum(P, sum);
     Compute_P_index_Sum(MaterialMask, count);
     P_av_z = sum/count;
}
