#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os.path as op


def qap_anatomical_spatial_workflow(workflow, resource_pool, config,
                                    plot_mask=False):
    from qap.workflows.anatomical import qap_anatomical_spatial_workflow
    wf = qap_anatomical_spatial_workflow(workflow, config, plot_mask)

    # Connect principal input
    if 'anatomical_scan' in resource_pool.keys():
        wf.inputs.inputnode.anatomical_scan = resource_pool['anatomical_scan']

    # Connect possibly cached inputs
    if 'anatomical_reorient' in resource_pool.keys():
        wf.inputs.cachenode.anatomical_reorient = \
            resource_pool['anatomical_reorient']
    if 'anatomical_brain' in resource_pool.keys():
        wf.inputs.cachenode.anatomical_brain = \
            resource_pool['anatomical_brain']
    if 'qap_head_mask' in resource_pool.keys():
        wf.inputs.cachenode.head_mask_path = \
            resource_pool['qap_head_mask']

    if (('anatomical_gm_mask' in resource_pool.keys()) and
        ('anatomical_wm_mask' in resource_pool.keys()) and
            ('anatomical_csf_mask' in resource_pool.keys())):
        wf.inputs.cachenode.anatomical_gm_mask = \
            resource_pool['anatomical_gm_mask']
        wf.inputs.cachenode.anatomical_wm_mask = \
            resource_pool['anatomical_wm_mask']
        wf.inputs.cachenode.inputs.anatomical_csf_mask = \
            resource_pool['anatomical_csf_mask']

    # Maintain backwards compatibility with resource_pool
    resource_pool['qap_anatomical_spatial'] = (
        wf.get_node('qap_anatomical_spatial_to_csv'), 'csv_file')

    if config.get('write_report', False):
        resource_pool['qap_anatomical_mosaic'] = (
            wf.get_node('plot_mosaic'), 'out_file')

    return wf, resource_pool


def qap_functional_spatial_workflow(workflow, resource_pool, config,
                                    plot_mask=False):
    from qap.workflows.functional_spatial import \
        qap_functional_spatial_workflow

    wf = qap_functional_spatial_workflow(workflow, config, plot_mask)

    # Connect principal input
    if 'functional_scan' in resource_pool.keys():
        wf.inputs.inputnode.anatomical_scan = resource_pool['functional_scan']

    # Connect possibly cached inputs
    if 'mean_functional' in resource_pool.keys():
        wf.inputs.cachenode.mean_functional = resource_pool['mean_functional']
    if 'functional_brain_mask' in resource_pool.keys():
        wf.inputs.cachenode.functional_brain_mask = \
            resource_pool['functional_brain_mask']

    # Maintain backwards compatibility with resource_pool
    resource_pool['qap_functional_spatial'] = (
        wf.get_node('qap_functional_spatial_to_csv'), 'csv_file')

    if config.get('write_report', False):
        resource_pool['qap_functional_mosaic'] = (
            wf.get_node('plot_mosaic'), 'out_file')

    return wf, resource_pool


def qap_functional_temporal_workflow(workflow, resource_pool, config):

    # resource pool should have:
    #     functional_brain_mask
    #     func_motion_correct
    #     coordinate_transformation

    import os
    import sys
    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe
    import nipype.interfaces.utility as niu
    import nipype.algorithms.misc as nam

    from qap.workflows.utils import qap_functional_temporal
    from temporal_qc import fd_jenkinson
    from qap.viz.interfaces import PlotMosaic, PlotFD

    def _getfirst(inlist):
        if isinstance(inlist, list):
            return inlist[0]

        return inlist

    # if 'mean_functional' not in resource_pool.keys():
    #     from functional_preproc import mean_functional_workflow
    #     workflow, resource_pool = \
    #         mean_functional_workflow(workflow, resource_pool, config)

    if 'functional_brain_mask' not in resource_pool.keys():
        from functional_preproc import functional_brain_mask_workflow
        workflow, resource_pool = \
            functional_brain_mask_workflow(workflow, resource_pool, config)

    if ('func_motion_correct' not in resource_pool.keys()) or \
        ('coordinate_transformation' not in resource_pool.keys() and
            'mcflirt_rel_rms' not in resource_pool.keys()):
        from functional_preproc import func_motion_correct_workflow
        workflow, resource_pool = \
            func_motion_correct_workflow(workflow, resource_pool, config)

    fd = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file'],
        function=fd_jenkinson), name='generate_FD_file')

    if 'mcflirt_rel_rms' in resource_pool.keys():
        fd.inputs.in_file = resource_pool['mcflirt_rel_rms']
    else:
        if len(resource_pool['coordinate_transformation']) == 2:
            node, out_file = resource_pool['coordinate_transformation']
            workflow.connect(node, out_file, fd, 'in_file')
        else:
            fd.inputs.in_file = resource_pool['coordinate_transformation']

    temporal = pe.Node(niu.Function(
        input_names=['func_motion_correct', 'func_brain_mask', 'tsnr_volume',
                     'fd_file', 'subject_id', 'session_id',
                     'scan_id', 'site_name'], output_names=['qc'],
        function=qap_functional_temporal), name='qap_functional_temporal')
    temporal.inputs.subject_id = config['subject_id']
    temporal.inputs.session_id = config['session_id']
    temporal.inputs.scan_id = config['scan_id']
    workflow.connect(fd, 'out_file', temporal, 'fd_file')

    if 'site_name' in config.keys():
        temporal.inputs.site_name = config['site_name']

    tsnr = pe.Node(nam.TSNR(), name='compute_tsnr')
    if len(resource_pool['func_motion_correct']) == 2:
        node, out_file = resource_pool['func_motion_correct']
        workflow.connect(node, out_file, tsnr, 'in_file')
        workflow.connect(node, out_file, temporal, 'func_motion_correct')
    else:
        from workflow_utils import check_input_resources
        check_input_resources(resource_pool, 'func_motion_correct')
        input_file = resource_pool['func_motion_correct']
        tsnr.inputs.in_file = input_file
        temporal.inputs.func_motion_correct = input_file

    if len(resource_pool['functional_brain_mask']) == 2:
        node, out_file = resource_pool['functional_brain_mask']
        workflow.connect(node, out_file, temporal, 'func_brain_mask')
    else:
        temporal.inputs.func_brain_mask = \
            resource_pool['functional_brain_mask']

    # Write mosaic and FD plot
    if config.get('write_report', False):
        plot = pe.Node(PlotMosaic(), name='plot_mosaic')
        plot.inputs.subject = config['subject_id']

        metadata = [config['session_id'], config['scan_id']]
        if 'site_name' in config.keys():
            metadata.append(config['site_name'])

        plot.inputs.metadata = metadata
        plot.inputs.title = 'tSNR volume'
        workflow.connect(tsnr, 'tsnr_file', plot, 'in_file')

        # Enable this if we want masks
        # if len(resource_pool['functional_brain_mask']) == 2:
        #     node, out_file = resource_pool['functional_brain_mask']
        #     workflow.connect(node, out_file, plot, 'in_mask')
        # else:
        #     plot.inputs.in_mask = resource_pool['functional_brain_mask']
        resource_pool['qap_mosaic'] = (plot, 'out_file')

        fdplot = pe.Node(PlotFD(), name='plot_fd')
        fdplot.inputs.subject = config['subject_id']
        fdplot.inputs.metadata = metadata
        workflow.connect(fd, 'out_file', fdplot, 'in_file')
        resource_pool['qap_fd'] = (fdplot, 'out_file')

    out_csv = op.join(
        config['output_directory'], 'qap_functional_temporal.csv')
    temporal_to_csv = pe.Node(
        nam.AddCSVRow(in_file=out_csv), name='qap_functional_temporal_to_csv')

    workflow.connect(tsnr, 'tsnr_file', temporal, 'tsnr_volume')
    workflow.connect(temporal, 'qc', temporal_to_csv, '_outputs')
    resource_pool['qap_functional_temporal'] = (temporal_to_csv, 'csv_file')
    return workflow, resource_pool
