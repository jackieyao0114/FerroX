#include "FerroX.H"
#include "AMReX_PlotFileUtil.H"
#include "Input/GeometryProperties/GeometryProperties.H"

void WritePlotfile(c_FerroX& rFerroX,
                   MultiFab& PoissonPhi,
                   MultiFab& PoissonRHS,
                   Array< MultiFab, AMREX_SPACEDIM>& P_old,
                   Array< MultiFab, AMREX_SPACEDIM>& E,
                   MultiFab& hole_den,
                   MultiFab& e_den,
                   MultiFab& charge_den,
                   MultiFab& beta_cc,
                   MultiFab& MaterialMask,
                   MultiFab& tphaseMask,
                   MultiFab& BigGamma,
                   MultiFab& alpha, 
                   MultiFab& beta, 
                   MultiFab& gamma, 
                   MultiFab& epsilonX_fe, 
                   MultiFab& epsilonZ_fe, 
                   MultiFab& epsilon_de, 
                   MultiFab& epsilon_si, 
                   MultiFab& g11, 
                   MultiFab& g44, 
                   MultiFab& g44_p, 
                   MultiFab& g12, 
                   MultiFab& alpha_12, 
                   MultiFab& alpha_112, 
                   MultiFab& alpha_123,
                   MultiFab& angle_alpha,
                   MultiFab& angle_beta,
                   MultiFab& angle_theta,
                   MultiFab& Phidiff,
                   const Geometry& geom,
                   const Real& time,
                   const int& plt_step)
{
    // timer for profiling
    BL_PROFILE_VAR("WritePlotfile()",WritePlotfile);

    BoxArray ba = PoissonPhi.boxArray();
    DistributionMapping dm = PoissonPhi.DistributionMap();

    const std::string& pltfile = amrex::Concatenate("plt",plt_step,8);

    Vector<std::string> var_names;

    //Px, Py, Pz
    int nvar = 3;

    var_names.push_back("Px");
    var_names.push_back("Py");
    var_names.push_back("Pz");

    if (plot_Phi) {
        ++nvar;
        var_names.push_back("Phi");
    }

    if (plot_PoissonRHS) {
        ++nvar;
        var_names.push_back("PoissonRHS");
    }

    if (plot_E) {
        nvar += 3;
        var_names.push_back("Ex");
        var_names.push_back("Ey");
        var_names.push_back("Ez");
    }

    if (plot_holes) {
        ++nvar;
        var_names.push_back("holes");
    }

    if (plot_electrons) {
        ++nvar;
        var_names.push_back("electrons");
    }

    if (plot_charge) {
        ++nvar;
        var_names.push_back("charge");
    }

    if (plot_epsilon) {
        ++nvar;
        var_names.push_back("epsilon");
    }

    if (plot_mask) {
        ++nvar;
        var_names.push_back("mask");
    }

    if (plot_tphase) {
        ++nvar;
        var_names.push_back("tphase");
    }

    if (plot_mat_BigGamma) {
        ++nvar;
        var_names.push_back("BigGamma");
    }
    if (plot_mat_alpha) {
        ++nvar;
        var_names.push_back("alpha");
    }
    if (plot_mat_beta) {
        ++nvar;
        var_names.push_back("beta");
    }
    if (plot_mat_gamma) {
        ++nvar;
        var_names.push_back("gamma");
    }
    if (plot_mat_epsilonX_fe) {
        ++nvar;
        var_names.push_back("epsilonX_fe");
    }
    if (plot_mat_epsilonZ_fe) {
        ++nvar;
        var_names.push_back("epsilonZ_fe");
    }
    if (plot_mat_epsilon_de) {
        ++nvar;
        var_names.push_back("epsilon_de");
    }
    if (plot_mat_epsilon_si) {
        ++nvar;
        var_names.push_back("epsilon_si");
    }
    if (plot_mat_g11) {
        ++nvar;
        var_names.push_back("g11");
    }
    if (plot_mat_g44) {
        ++nvar;
        var_names.push_back("g44");
    }
    if (plot_mat_g44_p) {
        ++nvar;
        var_names.push_back("g44_p");
    }
    if (plot_mat_g12) {
        ++nvar;
        var_names.push_back("g12");
    }
    if (plot_mat_alpha_12) {
        ++nvar;
        var_names.push_back("alpha_12");
    }
    if (plot_mat_alpha_112) {
        ++nvar;
        var_names.push_back("alpha_112");
    }
    if (plot_mat_alpha_123) {
        ++nvar;
        var_names.push_back("alpha_123");
    }

    if (plot_angle_alpha) {
        ++nvar;
        var_names.push_back("angle_alpha");
    }

    if (plot_angle_beta) {
        ++nvar;
        var_names.push_back("angle_beta");
    }

    if (plot_angle_theta) {
        ++nvar;
        var_names.push_back("angle_theta");
    }

    if (plot_PhiDiff) {
        ++nvar;
        var_names.push_back("PhiDiff");
    }



    auto& rGprop = rFerroX.get_GeometryProperties();
#ifdef AMREX_USE_EB
    MultiFab Plt(ba, dm, nvar, 0,  MFInfo(), *rGprop.pEB->p_factory_union);
#else    
    MultiFab Plt(ba, dm, nvar, 0);
#endif

    int counter = 0;

    MultiFab::Copy(Plt, P_old[0], 0, counter++, 1, 0);
    MultiFab::Copy(Plt, P_old[1], 0, counter++, 1, 0);
    MultiFab::Copy(Plt, P_old[2], 0, counter++, 1, 0);  

    if (plot_Phi) {
        MultiFab::Copy(Plt, PoissonPhi, 0, counter++, 1, 0);
    }

    if (plot_PoissonRHS) {
        MultiFab::Copy(Plt, PoissonRHS, 0, counter++, 1, 0);
    }

    if (plot_E) {
        MultiFab::Copy(Plt, E[0], 0, counter++, 1, 0);
        MultiFab::Copy(Plt, E[1], 0, counter++, 1, 0);
        MultiFab::Copy(Plt, E[2], 0, counter++, 1, 0);  
    }

    if (plot_holes) {
        MultiFab::Copy(Plt, hole_den, 0, counter++, 1, 0);
    }

    if (plot_electrons) {
        MultiFab::Copy(Plt, e_den, 0, counter++, 1, 0);
    }

    if (plot_charge) {
        MultiFab::Copy(Plt, charge_den, 0, counter++, 1, 0);
    }

    if (plot_epsilon) {
        MultiFab::Copy(Plt, beta_cc, 0, counter++, 1, 0);
    }

    if (plot_mask) {
        MultiFab::Copy(Plt, MaterialMask, 0, counter++, 1, 0);
    }

    if (plot_tphase) {
        MultiFab::Copy(Plt, tphaseMask, 0, counter++, 1, 0);
    }

    if (plot_mat_BigGamma) {
        MultiFab::Copy(Plt, BigGamma, 0, counter++, 1, 0);
    }
    if (plot_mat_alpha) {
        MultiFab::Copy(Plt, alpha, 0, counter++, 1, 0);
    }
    if (plot_mat_beta) {
        MultiFab::Copy(Plt, beta, 0, counter++, 1, 0);
    }
    if (plot_mat_gamma) {
        MultiFab::Copy(Plt, gamma, 0, counter++, 1, 0);
    }
    if (plot_mat_epsilonX_fe) {
        MultiFab::Copy(Plt, epsilonX_fe, 0, counter++, 1, 0);
    }
    if (plot_mat_epsilonZ_fe) {
        MultiFab::Copy(Plt, epsilonZ_fe, 0, counter++, 1, 0);
    }
    if (plot_mat_epsilon_de) {
        MultiFab::Copy(Plt, epsilon_de, 0, counter++, 1, 0);
    }
    if (plot_mat_epsilon_si) {
        MultiFab::Copy(Plt, epsilon_si, 0, counter++, 1, 0);
    }
    if (plot_mat_g11) {
        MultiFab::Copy(Plt, g11, 0, counter++, 1, 0);
    }
    if (plot_mat_g44) {
        MultiFab::Copy(Plt, g44, 0, counter++, 1, 0);
    }
    if (plot_mat_g44_p) {
        MultiFab::Copy(Plt, g44_p, 0, counter++, 1, 0);
    }
    if (plot_mat_g12) {
        MultiFab::Copy(Plt, g12, 0, counter++, 1, 0);
    }
    if (plot_mat_alpha_12) {
        MultiFab::Copy(Plt, alpha_12, 0, counter++, 1, 0);
    }
    if (plot_mat_alpha_112) {
        MultiFab::Copy(Plt, alpha_12, 0, counter++, 1, 0);
    }
    if (plot_mat_alpha_123) {
        MultiFab::Copy(Plt, alpha_123, 0, counter++, 1, 0);
    }

    if (plot_angle_alpha) {
        MultiFab::Copy(Plt, angle_alpha, 0, counter++, 1, 0);
    }

    if (plot_angle_beta) {
        MultiFab::Copy(Plt, angle_beta, 0, counter++, 1, 0);
    }

    if (plot_angle_theta) {
        MultiFab::Copy(Plt, angle_theta, 0, counter++, 1, 0);
    }

    if (plot_PhiDiff) {
        MultiFab::Copy(Plt, Phidiff, 0, counter++, 1, 0);
    }

    WriteSingleLevelPlotfile(pltfile, Plt, var_names, geom, time, plt_step);
}