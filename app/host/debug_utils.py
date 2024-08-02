from host import tasks
from host.base_tasks import TransientTaskRunner
from host.models import *
from host.transient_tasks import *
from host.system_tasks import *
import numpy as np


def _overwrite_or_create_object(model, unique_object_query, object_data):
    """
    Overwrites or creates new objects in the blast database.

    Parameters
        model (dango.model): blast model of the object that needs to be updated
        unique_object_query (dict): query to be passed to model.objects.get that will
            uniquely identify the object of interest
        object_data (dict): data to be saved or overwritten for the object.
    """

    try:
        object = model.objects.get(**unique_object_query)
        object.delete()
        model.objects.create(**object_data)
    except model.DoesNotExist:
        model.objects.create(**object_data)


def get_failed_tasks(transient_name=None):
    if transient_name is None:
        failed_task_register = TaskRegister.objects.filter(status__message="failed")
    else:
        transient = Transient.objects.get(name=transient_name)
        failed_task_register = TaskRegister.objects.filter(
            transient=transient, status__message="failed"
        )

    return failed_task_register


def rerun_failed_task(task_register):

    periodic_tasks = [
        MWEBV_Transient(task_register.transient.name),
        Ghost(task_register.transient.name),
        MWEBV_Host(task_register.transient.name),
        ImageDownload(task_register.transient.name),
        GlobalApertureConstruction(task_register.transient.name),
        LocalAperturePhotometry(task_register.transient.name),
        GlobalAperturePhotometry(task_register.transient.name),
        ValidateLocalPhotometry(task_register.transient.name),
        ValidateGlobalPhotometry(task_register.transient.name),
        TransientInformation(task_register.transient.name),
        HostInformation(task_register.transient.name),
        GlobalHostSEDFitting(task_register.transient.name),
        LocalHostSEDFitting(task_register.transient.name),
        TNSDataIngestion(),
        InitializeTransientTasks(),
        IngestMissedTNSTransients(),
        DeleteGHOSTFiles(),
        SnapshotTaskRegister()
    ]

    
    task = task_register.task
    for ptask in periodic_tasks:
        if ptask.task_name == task.name:
            print(f"Running {task.name}")
            status = ptask._run_process(task_register.transient)
            print(f"Status: {status}")
    s = Status.objects.get(message=status)
    task_register.status = s
    task_register.save()
    return status


def set_tasks_unprocessed(transient_name):
    transient = Transient.objects.get(name=transient_name)
    all_tasks = TaskRegister.objects.filter(transient=transient)
    for t in all_tasks:
        s = Status.objects.get(message="not processed")
        t.status = s
        t.save()

def populate_sed_quantities():

    from host import postprocess_prosp as pp
    from host.prospector import build_model
    from host.prospector import build_obs
    import h5py
    import prospect.io.read_results as reader
    
    seds = SEDFittingResult.objects.all()
    for s in seds:
        if s.dust_index_50 is not None:
            continue

        percentiles = np.load(s.percentiles_file,allow_pickle=True)
        perc = np.atleast_1d(percentiles["percentiles"])[0]
        res, _, _ = reader.results_from(s.posterior.name, dangerous=False)
        
        
        logmass16, logmass50, logmass84 = perc["stellar_mass"]
        age16, age50, age84 = perc["mwa"]
        logsfr16, logsfr50, logsfr84 = np.log10(perc["sfr"][0])
        logssfr16, logssfr50, logssfr84 = np.log10(perc["ssfr"][0])
        logzsol16,logzsol50,logzsol84 = perc['logzsol']
        dust2_16,dust2_50,dust2_84 = perc['dust2']
        dust_index_16,dust_index_50,dust_index_84 = perc['dust_index']
        dust1_fraction_16,dust1_fraction_50,dust1_fraction_84 = perc['dust1_fraction']
        log_fagn_16,log_fagn_50,log_fagn_84 = perc['log_fagn']
        log_agn_tau_16,log_agn_tau_50,log_agn_tau_84 = perc['log_agn_tau']
        gas_logz_16,gas_logz_50,gas_logz_84 = perc['gas_logz']
        duste_qpah_16,duste_qpah_50,duste_qpah_84 = perc['duste_qpah']
        duste_umin_16,duste_umin_50,duste_umin_84 = perc['duste_umin']
        log_duste_gamma_16,log_duste_gamma_50,log_duste_gamma_84 = perc['log_duste_gamma']

        agebins = pp.z_to_agebins(s.transient.best_redshift)
        agebins_ago = 10**agebins / 1e9

        theta_index = pp.theta_index()
        sfh_binned = pp.getSFH(res['chain'],zred=s.transient.best_redshift,theta_index=theta_index,rtn_chains=True)[-1]
        sfh_binned = np.percentile(sfh_binned, [15.9,50,84.1], axis=0).T

        sfh_results = []
        unique_sfh = np.unique(perc['sfh'][:,1])
        for i,a,sf in zip(range(len(agebins_ago)),agebins_ago,sfh_binned):
            sfh_results += [
                {
                    'transient':s.transient,
                    'aperture':s.aperture,
                    'logsfr_16':np.log10(sf[0]),
                    'logsfr_50':np.log10(sf[1]),
                    'logsfr_84':np.log10(sf[2]),
                    'logsfr_tmin':a[0],
                    'logsfr_tmax':a[1]
                }
            ]

        prosp_results = {
            "transient": s.transient,
            "aperture": s.aperture,
            "log_mass_16": logmass16,
            "log_mass_50": logmass50,
            "log_mass_84": logmass84,
            "log_sfr_16": logsfr16,
            "log_sfr_50": logsfr50,
            "log_sfr_84": logsfr84,
            "log_ssfr_16": logssfr16,
            "log_ssfr_50": logssfr50,
            "log_ssfr_84": logssfr84,
            "log_age_16": age16,
            "log_age_50": age50,
            "log_age_84": age84,
            "logzsol_16":logzsol16,
            "logzsol_50":logzsol50,
            "logzsol_84":logzsol84,
            "dust2_16":dust2_16,
            "dust2_50":dust2_50,
            "dust2_84":dust2_84,
            "dust_index_16":dust_index_16,
            "dust_index_50":dust_index_50,
            "dust_index_84":dust_index_84,
            "dust1_fraction_16":dust1_fraction_16,
            "dust1_fraction_50":dust1_fraction_50,
            "dust1_fraction_84":dust1_fraction_84,
            "log_fagn_16":log_fagn_16,
            "log_fagn_50":log_fagn_50,
            "log_fagn_84":log_fagn_84,
            "log_agn_tau_16":log_agn_tau_16,
            "log_agn_tau_50":log_agn_tau_50,
            "log_agn_tau_84":log_agn_tau_84,
            "gas_logz_16":gas_logz_16,
            "gas_logz_50":gas_logz_50,
            "gas_logz_84":gas_logz_84,
            "duste_qpah_16":duste_qpah_16,
            "duste_qpah_50":duste_qpah_50,
            "duste_qpah_84":duste_qpah_84,
            "duste_umin_16":duste_umin_16,
            "duste_umin_50":duste_umin_50,
            "duste_umin_84":duste_umin_84,
            "log_duste_gamma_16":log_duste_gamma_16,
            "log_duste_gamma_50":log_duste_gamma_50,
            "log_duste_gamma_84":log_duste_gamma_84,
        }

        SEDFittingResult.objects.filter(pk=s.pk).update(**prosp_results)
        for sfh_r in sfh_results:
            ps = s.logsfh.filter(
                logsfr_tmin=sfh_r['logsfr_tmin']
            )
            sfh_r['transient'] = transient
            sfh_r['aperture'] = aperture[0]
            if len(ps):
                ps.update(**sfh_r)
            else:
                ps = StarFormationHistoryResult.objects.create(**sfh_r)
                s.logsfh.add(ps)
