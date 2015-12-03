#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:


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
        resource_pool['qap_mosaic'] = (
            wf.get_node('plot_mosaic'), 'out_file')

    return wf, resource_pool


def qap_functional_spatial_workflow(workflow, resource_pool, config,
                                    plot_mask=False):
    from qap.workflows.functional_spatial import \
        qap_functional_spatial_workflow

    wf = qap_functional_spatial_workflow(workflow, config, plot_mask)

    # Connect principal input
    if 'functional_scan' in resource_pool.keys():
        wf.inputs.inputnode.functional_scan = resource_pool['functional_scan']

    # Connect possibly cached inputs
    if 'mean_functional' in resource_pool.keys():
        wf.inputs.cachenode.mean_functional = resource_pool['mean_functional']
    if 'functional_brain_mask' in resource_pool.keys():
        wf.inputs.cachenode.functional_brain_mask = \
            resource_pool['functional_brain_mask']

    # Subject infos
    wf.inputs.inputnode.subject_id = config['subject_id']
    wf.inputs.inputnode.session_id = config['session_id']
    wf.inputs.inputnode.scan_id = config['scan_id']
    wf.inputs.inputnode.direction = config.get('ghost_direction', 'y')
    wf.inputs.inputnode.start_idx = config.get('start_idx', 0)
    wf.inputs.inputnode.stop_idx = config.get('stop_idx', None)

    if 'site_name' in config.keys():
        wf.inputs.inputnode.site_name = config['site_name']

    # Maintain backwards compatibility with resource_pool
    resource_pool['qap_functional_spatial'] = (
        wf.get_node('qap_functional_spatial_to_csv'), 'csv_file')

    if config.get('write_report', False):
        resource_pool['qap_mosaic'] = (
            wf.get_node('plot_mosaic'), 'out_file')

    return wf, resource_pool


def qap_functional_temporal_workflow(workflow, resource_pool, config,
                                     plot_mask=False):
    from qap.workflows.functional_temporal import \
        qap_functional_temporal_workflow

    wf = qap_functional_temporal_workflow(workflow, config, plot_mask)

    # Connect principal input
    if 'functional_scan' in resource_pool.keys():
        wf.inputs.inputnode.functional_scan = resource_pool['functional_scan']

    # Subject infos
    wf.inputs.inputnode.subject_id = config['subject_id']
    wf.inputs.inputnode.session_id = config['session_id']
    wf.inputs.inputnode.scan_id = config['scan_id']
    wf.inputs.inputnode.direction = config.get('ghost_direction', 'y')
    wf.inputs.inputnode.start_idx = config.get('start_idx', 0)
    wf.inputs.inputnode.stop_idx = config.get('stop_idx', None)

    if 'site_name' in config.keys():
        wf.inputs.inputnode.site_name = config['site_name']

    # Connect possibly cached inputs
    if 'functional_brain_mask' in resource_pool.keys():
        wf.inputs.cachenode.functional_brain_mask = \
            resource_pool['functional_brain_mask']
    if 'func_motion_correct' in resource_pool.keys():
        wf.inputs.cachenode.func_motion_correct = \
            resource_pool['func_motion_correct']
        wf.inputs.cachenode.coordinate_transformation = \
            resource_pool['coordinate_transformation']
        wf.inputs.cachenode.mcflirt_rel_rms = \
            resource_pool['mcflirt_rel_rms']

    # Maintain backwards compatibility with resource_pool
    resource_pool['qap_functional_temporal'] = (
        wf.get_node('qap_functional_temporal_to_csv'), 'csv_file')

    if config.get('write_report', False):
        resource_pool['qap_mosaic'] = (
            wf.get_node('plot_mosaic'), 'out_file')

    return wf, resource_pool
