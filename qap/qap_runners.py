#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os.path as op
from qap_workflows import qap_mask_workflow, qap_anatomical_spatial_workflow, \
    qap_functional_spatial_workflow, qap_functional_temporal_workflow


def run_qap_mask(anatomical_reorient, flirt_affine_xfm, template_skull,
                 run=True):

    # stand-alone runner for anatomical reorient workflow

    import os
    import sys

    import glob

    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe

    output = 'qap_head_mask'

    workflow = pe.Workflow(name='%s_workflow' % output)

    current_dir = os.getcwd()

    workflow_dir = os.path.join(current_dir, output)
    workflow.base_dir = workflow_dir

    resource_pool = {}
    config = {}
    num_cores_per_subject = 1

    resource_pool['anatomical_reorient'] = anatomical_reorient
    resource_pool['flirt_affine_xfm'] = flirt_affine_xfm
    config['template_skull_for_anat'] = template_skull

    workflow, resource_pool = \
        qap_mask_workflow(workflow, resource_pool, config)

    ds = pe.Node(nio.DataSink(), name='datasink_%s' % output)
    ds.inputs.base_directory = workflow_dir

    node, out_file = resource_pool[output]

    workflow.connect(node, out_file, ds, output)

    if run:
        workflow.run(
            plugin='MultiProc', plugin_args={'n_procs': num_cores_per_subject})
        outpath = glob.glob(os.path.join(workflow_dir, output, '*'))[0]
        return outpath

    else:
        return workflow, workflow.base_dir


def run_single_qap_anatomical_spatial(
        anatomical_reorient, qap_head_mask, anatomical_csf_mask,
        anatomical_gm_mask, anatomical_wm_mask, subject_id,
        session_id, scan_id, site_name=None, run=True):

    # stand-alone runner for anatomical spatial QAP workflow

    import os
    import sys
    import glob
    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe

    output = 'qap_anatomical_spatial'
    workflow = pe.Workflow(name='%s_workflow' % output)

    current_dir = os.getcwd()
    workflow_dir = os.path.join(current_dir, output)
    workflow.base_dir = workflow_dir

    num_cores_per_subject = 1
    resource_pool = {
        'anatomical_reorient': anatomical_reorient,
        'qap_head_mask': qap_head_mask,
        'anatomical_csf_mask': anatomical_csf_mask,
        'anatomical_gm_mask': anatomical_gm_mask,
        'anatomical_wm_mask': anatomical_wm_mask
    }

    config = {
        'subject_id': subject_id,
        'session_id': session_id,
        'scan_id': scan_id
    }

    if site_name:
        config['site_name'] = site_name

    workflow, resource_pool = \
        qap_anatomical_spatial_workflow(workflow, resource_pool, config)

    ds = pe.Node(nio.DataSink(), name='datasink_%s' % output)
    ds.inputs.base_directory = workflow_dir

    node, out_file = resource_pool[output]

    workflow.connect(node, out_file, ds, output)

    if run:
        workflow.run(
            plugin='MultiProc', plugin_args={'n_procs': num_cores_per_subject})
        outpath = glob.glob(os.path.join(workflow_dir, output, '*'))[0]
        return outpath

    else:
        return workflow, workflow.base_dir


def run_single_qap_functional_spatial(
        mean_functional, functional_brain_mask, subject_id, session_id,
        scan_id, site_name=None, ghost_direction=None, run=True):

    # stand-alone runner for functional spatial QAP workflow
    import os
    import sys
    import glob
    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe

    output = 'qap_functional_spatial'
    workflow = pe.Workflow(name='%s_workflow' % output)
    current_dir = os.getcwd()
    workflow_dir = os.path.join(current_dir, output)
    workflow.base_dir = workflow_dir

    resource_pool = {}
    config = {}
    num_cores_per_subject = 1

    resource_pool['mean_functional'] = mean_functional
    resource_pool['functional_brain_mask'] = functional_brain_mask

    config['subject_id'] = subject_id
    config['session_id'] = session_id
    config['scan_id'] = scan_id

    if site_name:
        config['site_name'] = site_name

    if ghost_direction:
        config['ghost_direction'] = ghost_direction

    workflow, resource_pool = \
        qap_functional_spatial_workflow(workflow, resource_pool, config)

    ds = pe.Node(nio.DataSink(), name='datasink_%s' % output)
    ds.inputs.base_directory = workflow_dir

    node, out_file = resource_pool[output]

    workflow.connect(node, out_file, ds, output)

    if run:
        workflow.run(
            plugin='MultiProc', plugin_args={'n_procs': num_cores_per_subject})
        outpath = glob.glob(os.path.join(workflow_dir, output, '*'))[0]
        return outpath

    else:
        return workflow, workflow.base_dir


def run_single_qap_functional_temporal(func_motion, functional_brain_mask,
                                       subject_id, session_id, scan_id,
                                       site_name=None, mcflirt_rel_rms=None,
                                       coordinate_transformation=None,
                                       run=True):

    # stand-alone runner for functional temporal QAP workflow

    import os
    import sys

    import glob

    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe

    output = 'qap_functional_temporal'

    workflow = pe.Workflow(name='%s_workflow' % output)

    current_dir = os.getcwd()

    workflow_dir = os.path.join(current_dir, output)
    workflow.base_dir = workflow_dir

    resource_pool = {}
    config = {}
    num_cores_per_subject = 1

    resource_pool['func_motion_correct'] = func_motion
    resource_pool['functional_brain_mask'] = functional_brain_mask

    if mcflirt_rel_rms:
        resource_pool['mcflirt_rel_rms'] = mcflirt_rel_rms
    elif coordinate_transformation:
        resource_pool['coordinate_transformation'] = coordinate_transformation

    config['subject_id'] = subject_id
    config['session_id'] = session_id
    config['scan_id'] = scan_id

    if site_name:
        config['site_name'] = site_name

    workflow, resource_pool = \
        qap_functional_temporal_workflow(workflow, resource_pool, config)

    ds = pe.Node(nio.DataSink(), name='datasink_%s' % output)
    ds.inputs.base_directory = workflow_dir

    node, out_file = resource_pool[output]

    workflow.connect(node, out_file, ds, output)

    if run:
        workflow.run(
            plugin='MultiProc', plugin_args={'n_procs': num_cores_per_subject})

        outpath = glob.glob(os.path.join(workflow_dir, output, '*'))[0]

        return outpath

    else:
        return workflow, workflow.base_dir
