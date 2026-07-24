"""
This module compares the tabular transient data exported by the /api/transient/export/[transient] endpoint,
detailing differences between the data structures.

Example usage:

LOG_LEVEL=WARNING python validation/validation_scripts/compare_results.py \
    ~/2026dix.v1.2.3.json \
    ~/2026dix.v1.3.0.json

"""

import json
import sys
import os
import logging

# Configure logging
logging.basicConfig(format='%(asctime)s [%(name)-8s] %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', logging.DEBUG))


def compare_vals(val1, val2, key, label, percent_diff=2.0):
    if isinstance(val1, float) and abs(val1 - val2) < abs(val1 * percent_diff / 100.0):
        logger.debug(f'''[{label}] match within {percent_diff}% [{key}]: {val1} == {val2}''')
    elif (isinstance(val1, float) and key.endswith('_deg')
            and (360.0 - abs(val1 - val2)) / 360.0 <= percent_diff / 100.0):
        logger.debug(f'''[{label}] angular match within {percent_diff}% : {val1} == {val2}''')
    elif val1 == val2:
        logger.debug(f'''[{label}] exact match [{key}]: {val1} == {val2}''')
    else:
        numeric_mismatch = f''' over {percent_diff}%''' if not isinstance(val1, str) and val1 is not None else ''
        logger.warning(f'''[{label}] mismatch{numeric_mismatch} [{key}]:''')
        logger.warning(f'''    val1: "{val1}"''')
        logger.warning(f'''    val2: "{val2}"''')


def compare_data(t1, t2):
    # Filters & surveys
    for t1_filter in t1['filters']:
        filter_name = t1_filter['fields']['name']
        # Identify t2 filter by name
        t2_filter = [f for f in t2['filters'] if f['fields']['name'] == filter_name][0]
        # Compare associated surveys
        t1_survey = [s for s in t1['surveys'] if s['pk'] == t1_filter['fields']['survey']][0]
        t2_survey = [s for s in t2['surveys'] if s['pk'] == t2_filter['fields']['survey']][0]
        try:
            t1_survey_name = t1_survey['fields']['name']
            t2_survey_name = t2_survey['fields']['name']
            assert t1_survey['fields']['name'] == t2_survey['fields']['name']
        except AssertionError:
            logger.warning(f'''Survey mismatch for filter "{filter_name}": "{t1_survey_name}" != "{t2_survey_name}"''')
            continue
        for key, val in t1_filter['fields'].items():
            # Ignored keys
            if key in ['software_version', 'survey']:
                continue
            val1 = val
            val2 = t2_filter['fields'][key]
            compare_vals(val1, val2, key, label=f'filter "{filter_name}"')
    # Transient
    for key, val in t1['transient']['fields'].items():
        # Ignored keys
        if key in ['software_version', 'host']:
            continue
        val1 = val
        val2 = t2['transient']['fields'][key]
        compare_vals(val1, val2, key, label='transient')
    # Host
    for key, val in t1['host']['fields'].items():
        val1 = val
        val2 = t2['host']['fields'][key]
        compare_vals(val1, val2, key, label='host')

    # Cutouts
    for t1_cutout in t1['cutouts']:
        cutout_name = t1_cutout['fields']['name']
        # Identify t2 cutout by name
        t2_cutout = [c for c in t2['cutouts'] if c['fields']['name'] == cutout_name][0]
        # Compare associated filters
        t1_filter = [s for s in t1['filters'] if s['pk'] == t1_cutout['fields']['filter']][0]
        t2_filter = [s for s in t2['filters'] if s['pk'] == t2_cutout['fields']['filter']][0]
        try:
            t1_filter_name = t1_filter['fields']['name']
            t2_filter_name = t2_filter['fields']['name']
            assert t1_filter['fields']['name'] == t2_filter['fields']['name']
            logger.debug(f'''filter match for cutout "{cutout_name}": "{t1_filter_name}" == "{t2_filter_name}"''')
        except AssertionError:
            logger.warning(f'''filter mismatch for cutout "{cutout_name}": "{t1_filter_name}" != "{t2_filter_name}"''')
            continue
        for key, val in t1_cutout['fields'].items():
            # Ignored keys
            if key in ['software_version', 'filter', 'transient']:
                continue
            val1 = val
            val2 = t2_cutout['fields'][key]
            compare_vals(val1, val2, key, label=f'cutout "{cutout_name}"')

    # Apertures
    for t1_aperture in t1['apertures']:
        aperture_name = t1_aperture['fields']['name']
        t2_aperture = [a for a in t2['apertures'] if a['fields']['name'] == aperture_name][0]
        for key, val in t1_aperture['fields'].items():
            # Ignored keys
            if key in ['software_version', 'cutout', 'transient']:
                continue
            val1 = val
            val2 = t2_aperture['fields'][key]
            compare_vals(val1, val2, key, label=f'aperture "{aperture_name}"')
        for idx, t1_sed in enumerate(t1_aperture['sedfittingresults']):
            try:
                t2_aperture['sedfittingresults'][idx]
            except IndexError:
                logger.warning(f'''[sedfittingresults "{aperture_name}"] mismatch for sedfittingresults[{idx}]''')
                continue
            for key, val in t1_sed['fields'].items():
                # Ignored keys
                if key in ['software_version', 'aperture', 'transient', 'logsfh']:
                    continue
                val1 = val
                val2 = t2_aperture['sedfittingresults'][idx]['fields'][key]
                compare_vals(val1, val2, key, label=f'sedfittingresults "{aperture_name}"')
            # Relying on the ordering of the SFH result lists
            t2_sfh_results = []
            for sfh_result_idx in t2_aperture['sedfittingresults'][idx]['fields']['logsfh']:
                t2_sfh_results.extend([sfh for sfh in t2_aperture['starformationhistoryresult']
                                       if sfh['pk'] == sfh_result_idx])
            t1_sfh_results = []
            for sfh_result_idx in t1_aperture['sedfittingresults'][idx]['fields']['logsfh']:
                t1_sfh_results.extend([sfh for sfh in t1_aperture['starformationhistoryresult']
                                       if sfh['pk'] == sfh_result_idx])
            for sfh_idx, t1_sfh_result in enumerate(t1_sfh_results):
                for key, val in t1_sfh_result['fields'].items():
                    # Ignored keys
                    if key in ['software_version', 'aperture', 'transient']:
                        continue
                    val1 = val
                    val2 = t2_sfh_results[sfh_idx]['fields'][key]
                    compare_vals(val1, val2, key, label=f'sfh_result idx {t1_sfh_results[sfh_idx]['pk']}')
        for t1_aperturephotometry in t1_aperture['aperturephotometry']:
            t1_filter = [f for f in t1['filters'] if f['pk'] == t1_aperturephotometry['fields']['filter']][0]
            t1_filter_name = t1_filter['fields']['name']
            # Find the t2 filter with the same name
            t2_filter = [f for f in t2['filters'] if f['fields']['name'] == t1_filter_name][0]
            # Find the t2_aperturephotometry object that associated with that filter
            t2_aperturephotometry = [ap for ap in t2_aperture['aperturephotometry']
                                     if ap['fields']['filter'] == t2_filter['pk']][0]
            for key, val in t1_aperturephotometry['fields'].items():
                # Ignored keys
                if key in ['software_version', 'aperture', 'transient', 'filter']:
                    continue
                val1 = val
                val2 = t2_aperturephotometry['fields'][key]
                compare_vals(val1, val2, key, label=f'aperturephotometry "{filter_name}"')


def main():
    files = {}
    data = {}
    files[0] = sys.argv[1]
    files[1] = sys.argv[2]
    logger.debug(f'''Comparing files: "{files[0]}" == "{files[1]}"''')

    for idx in [0, 1]:
        with open(files[idx], 'r') as fp:
            data[idx] = json.load(fp)
        # with open(f'''{files[idx]}.sorted.json''', 'w') as fp:
        #     json.dump(data[idx], fp, sort_keys=True, indent=2)

    compare_data(data[0], data[1])


if __name__ == "__main__":
    main()
