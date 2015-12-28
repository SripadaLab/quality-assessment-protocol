#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import os.path as op
import sys

import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu

from .functional_common import (functional_brain_mask_workflow,
                                mean_functional_workflow)
from nipype import logging
logger = logging.getLogger('workflow')


def qap_functional_spatial_workflow(config, plot_mask=False):
    import nipype.algorithms.misc as nam
    from utils import qap_functional_spatial
    from qap.viz.interfaces import PlotMosaic

    settings = ['subject_id', 'session_id', 'scan_id', 'site_name',
                'direction']

    workflow = pe.Workflow(name=config['scan_id'])
    workflow.base_dir = op.join(config['working_directory'],
                                config['subject_id'],
                                config['session_id'])

    # set up crash directory
    workflow.config['execution'] = \
        {'crashdump_dir': config["output_directory"]}

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['functional_scan', 'start_idx', 'stop_idx']+settings),
        name='inputnode')

    # Subject infos
    inputnode.inputs.subject_id = config['subject_id']
    inputnode.inputs.session_id = config['session_id']
    inputnode.inputs.scan_id = config['scan_id']
    inputnode.inputs.direction = config.get('ghost_direction', 'y')
    inputnode.inputs.start_idx = config.get('start_idx', 0)
    inputnode.inputs.stop_idx = config.get('stop_idx', None)

    if 'site_name' in config.keys():
        inputnode.inputs.site_name = config['site_name']

    # resource pool should have:
    cfields = ['mean_functional', 'func_motion_correct',
               'functional_brain_mask']
    cache = pe.Node(niu.IdentityInterface(fields=cfields), name='cachenode')

    spatial_epi = pe.Node(niu.Function(
        input_names=['mean_epi', 'func_brain_mask']+settings,
        output_names=['qc'], function=qap_functional_spatial),
        name='qap_functional_spatial')

    workflow.connect([(inputnode, spatial_epi, [(k, k) for k in settings])])

    # mean_functional_workflow
    mfw = mean_functional_workflow(
        slice_timing_correction=config.get('slice_timing_correction', False))
    workflow.connect([
        (inputnode, mfw, [
            ('functional_scan', 'inputnode.functional_scan'),
            ('start_idx', 'inputnode.start_idx'),
            ('stop_idx', 'inputnode.stop_idx')]),
        (cache, mfw, [
            ('mean_functional', 'cachenode.mean_functional'),
            ('func_motion_correct', 'cachenode.func_motion_correct')]),
        (mfw, spatial_epi,
            [('outputnode.mean_functional', 'mean_epi')]),
    ])

    # functional_brain_mask_workflow
    bmw = functional_brain_mask_workflow(
        use_bet=config.get('use_bet', False),
        slice_timing_correction=config.get('slice_timing_correction', False))
    workflow.connect([
        (inputnode, bmw, [
            ('functional_scan', 'inputnode.functional_scan'),
            ('start_idx', 'inputnode.start_idx'),
            ('stop_idx', 'inputnode.stop_idx')]),
        (cache, bmw, [
            ('functional_brain_mask', 'cachenode.functional_brain_mask')]),
        (bmw, spatial_epi, [
            ('outputnode.functional_brain_mask', 'func_brain_mask')])
    ])

    # Write CSV row
    out_csv = op.join(config['output_directory'], 'qap_functional_spatial.csv')
    to_csv = pe.Node(
        nam.AddCSVRow(in_file=out_csv), name='qap_functional_spatial_to_csv')
    workflow.connect(spatial_epi, 'qc', to_csv, '_outputs')

    # Append plot generation
    if config.get('write_report', False):
        plot = pe.Node(PlotMosaic(), name='plot_mosaic')
        plot.inputs.subject = config['subject_id']

        metadata = [config['session_id'], config['scan_id']]
        if 'site_name' in config.keys():
            metadata.append(config['site_name'])

        plot.inputs.metadata = metadata
        plot.inputs.title = 'Mean EPI'
        workflow.connect(mfw, 'outputnode.mean_functional',
                         plot, 'in_file')
        if plot_mask:
            workflow.connect(bmw, 'outputnode.functional_brain_mask',
                             plot, 'in_mask')

    return workflow


def qap_functional_temporal_workflow(workflow, config, plot_mask=False):
    import nipype.algorithms.misc as nam
    from utils import qap_functional_temporal
    from ..temporal_qc import fd_jenkinson
    from qap.viz.interfaces import PlotMosaic, PlotFD

    settings = ['subject_id', 'session_id', 'scan_id', 'site_name',
                'direction']

    workflow = pe.Workflow(name=config['scan_id'])
    workflow.base_dir = op.join(config['working_directory'],
                                config['subject_id'],
                                config['session_id'])

    # set up crash directory
    workflow.config['execution'] = \
        {'crashdump_dir': config["output_directory"]}

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['functional_scan', 'start_idx', 'stop_idx']+settings),
        name='inputnode')

    # Subject infos
    inputnode.inputs.subject_id = config['subject_id']
    inputnode.inputs.session_id = config['session_id']
    inputnode.inputs.scan_id = config['scan_id']
    inputnode.inputs.direction = config.get('ghost_direction', 'y')
    inputnode.inputs.start_idx = config.get('start_idx', 0)
    inputnode.inputs.stop_idx = config.get('stop_idx', None)

    if 'site_name' in config.keys():
        inputnode.inputs.site_name = config['site_name']

    # resource pool should have:
    cfields = ['func_motion_correct', 'functional_brain_mask',
               'coordinate_transformation']
    cache = pe.Node(niu.IdentityInterface(fields=cfields), name='cachenode')

    # functional_brain_mask_workflow
    bmw = functional_brain_mask_workflow(
        use_bet=config.get('use_bet', False),
        slice_timing_correction=config.get('slice_timing_correction', False))
    workflow.connect([
        (inputnode, bmw, [
            ('functional_scan', 'inputnode.functional_scan'),
            ('start_idx', 'inputnode.start_idx'),
            ('stop_idx', 'inputnode.stop_idx')]),
        (cache, bmw, [
            ('functional_brain_mask', 'cachenode.functional_brain_mask')]),
        (bmw, spatial_epi, [
            ('outputnode.functional_brain_mask', 'func_brain_mask')])
    ])

    hmcwf = func_motion_correct_workflow(
        slice_timing_correction=slice_timing_correction)
    workflow.connect([
        (inputnode, hmcwf, [
            ('functional_scan', 'inputnode.functional_scan'),
            ('start_idx', 'inputnode.start_idx'),
            ('stop_idx', 'inputnode.stop_idx')]),
        (cachenode, hmcwf, [(
            ('func_motion_correct', 'cachenode.func_motion_correct'))])
    ])

    temp_epi = pe.Node(niu.Function(
        input_names=['func_motion_correct', 'func_brain_mask', 'tsnr_volume',
                     'fd_file']+settings,
        output_names=['qc'], function=qap_functional_temporal),
        name='qap_functional_temporal')

    workflow.connect([(inputnode, spatial_epi, [(k, k) for k in settings])])
    # TODO Connect missing inputs

    fd = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file'],
        function=fd_jenkinson), name='generate_FD_file')
    # TODO Connect either mcflirt_rel_rms, coordinate_transformation (cached)
    # or coordinate_transformation (from hmcwf) TO in_file

    tsnr = pe.Node(nam.TSNR(), name='compute_tsnr')
    workflow.connect([
        (hmcwf, tsnr, [('outputnode.func_motion_correct', 'in_file')]),
        (tsnr, temp_epi, [('tsnr_file', 'tsnr_volume')])
    ])

    # Write CSV row
    out_csv = op.join(config['output_directory'],
                      'qap_functional_temporal.csv')
    to_csv = pe.Node(
        nam.AddCSVRow(in_file=out_csv), name='qap_functional_temporal_to_csv')
    workflow.connect(temp_epi, 'qc', to_csv, '_outputs')

    # Append plot generation
    if config.get('write_report', False):
        plot = pe.Node(PlotMosaic(), name='plot_mosaic')
        plot.inputs.subject = config['subject_id']

        metadata = [config['session_id'], config['scan_id']]
        if 'site_name' in config.keys():
            metadata.append(config['site_name'])

        plot.inputs.metadata = metadata
        plot.inputs.title = 'tSNR volume'
        workflow.connect(tsnr, 'tsnr_file', plot, 'in_file')

        if plot_mask:
            workflow.connect(bmw, 'outputnode.functional_brain_mask',
                             plot, 'in_mask')

        fdplot = pe.Node(PlotFD(), name='plot_fd')
        fdplot.inputs.subject = config['subject_id']
        fdplot.inputs.metadata = metadata
        workflow.connect(fd, 'out_file', fdplot, 'in_file')

    return workflow


# Keep this here for reference until the new workflow is finished
# def qap_functional_temporal_workflow(workflow, resource_pool, config):

#     # resource pool should have:
#     #     functional_brain_mask
#     #     func_motion_correct
#     #     coordinate_transformation

#     import os
#     import sys
#     import nipype.interfaces.io as nio
#     import nipype.pipeline.engine as pe
#     import nipype.interfaces.utility as niu
#     import nipype.algorithms.misc as nam

#     from qap.workflows.utils import qap_functional_temporal
#     from temporal_qc import fd_jenkinson
#     from qap.viz.interfaces import PlotMosaic, PlotFD

#     def _getfirst(inlist):
#         if isinstance(inlist, list):
#             return inlist[0]

#         return inlist

#     # if 'mean_functional' not in resource_pool.keys():
#     #     from functional_preproc import mean_functional_workflow
#     #     workflow, resource_pool = \
#     #         mean_functional_workflow(workflow, resource_pool, config)

#     if 'functional_brain_mask' not in resource_pool.keys():
#         from functional_preproc import functional_brain_mask_workflow
#         workflow, resource_pool = \
#             functional_brain_mask_workflow(workflow, resource_pool, config)

#     if ('func_motion_correct' not in resource_pool.keys()) or \
#         ('coordinate_transformation' not in resource_pool.keys() and
#             'mcflirt_rel_rms' not in resource_pool.keys()):
#         from functional_preproc import func_motion_correct_workflow
#         workflow, resource_pool = \
#             func_motion_correct_workflow(workflow, resource_pool, config)

#     fd = pe.Node(niu.Function(
#         input_names=['in_file'], output_names=['out_file'],
#         function=fd_jenkinson), name='generate_FD_file')

#     if 'mcflirt_rel_rms' in resource_pool.keys():
#         fd.inputs.in_file = resource_pool['mcflirt_rel_rms']
#     else:
#         if len(resource_pool['coordinate_transformation']) == 2:
#             node, out_file = resource_pool['coordinate_transformation']
#             workflow.connect(node, out_file, fd, 'in_file')
#         else:
#             fd.inputs.in_file = resource_pool['coordinate_transformation']

#     temporal = pe.Node(niu.Function(
#         input_names=['func_motion_correct', 'func_brain_mask', 'tsnr_volume',
#                      'fd_file', 'subject_id', 'session_id',
#                      'scan_id', 'site_name'], output_names=['qc'],
#         function=qap_functional_temporal), name='qap_functional_temporal')
#     temporal.inputs.subject_id = config['subject_id']
#     temporal.inputs.session_id = config['session_id']
#     temporal.inputs.scan_id = config['scan_id']
#     workflow.connect(fd, 'out_file', temporal, 'fd_file')

#     if 'site_name' in config.keys():
#         temporal.inputs.site_name = config['site_name']

#     tsnr = pe.Node(nam.TSNR(), name='compute_tsnr')
#     if len(resource_pool['func_motion_correct']) == 2:
#         node, out_file = resource_pool['func_motion_correct']
#         workflow.connect(node, out_file, tsnr, 'in_file')
#         workflow.connect(node, out_file, temporal, 'func_motion_correct')
#     else:
#         from workflow_utils import check_input_resources
#         check_input_resources(resource_pool, 'func_motion_correct')
#         input_file = resource_pool['func_motion_correct']
#         tsnr.inputs.in_file = input_file
#         temporal.inputs.func_motion_correct = input_file

#     if len(resource_pool['functional_brain_mask']) == 2:
#         node, out_file = resource_pool['functional_brain_mask']
#         workflow.connect(node, out_file, temporal, 'func_brain_mask')
#     else:
#         temporal.inputs.func_brain_mask = \
#             resource_pool['functional_brain_mask']

#     # Write mosaic and FD plot
#     if config.get('write_report', False):
#         plot = pe.Node(PlotMosaic(), name='plot_mosaic')
#         plot.inputs.subject = config['subject_id']

#         metadata = [config['session_id'], config['scan_id']]
#         if 'site_name' in config.keys():
#             metadata.append(config['site_name'])

#         plot.inputs.metadata = metadata
#         plot.inputs.title = 'tSNR volume'
#         workflow.connect(tsnr, 'tsnr_file', plot, 'in_file')

#         # Enable this if we want masks
#         # if len(resource_pool['functional_brain_mask']) == 2:
#         #     node, out_file = resource_pool['functional_brain_mask']
#         #     workflow.connect(node, out_file, plot, 'in_mask')
#         # else:
#         #     plot.inputs.in_mask = resource_pool['functional_brain_mask']
#         resource_pool['qap_mosaic'] = (plot, 'out_file')

#         fdplot = pe.Node(PlotFD(), name='plot_fd')
#         fdplot.inputs.subject = config['subject_id']
#         fdplot.inputs.metadata = metadata
#         workflow.connect(fd, 'out_file', fdplot, 'in_file')
#         resource_pool['qap_fd'] = (fdplot, 'out_file')

#     out_csv = op.join(
#         config['output_directory'], 'qap_functional_temporal.csv')
#     temporal_to_csv = pe.Node(
#         nam.AddCSVRow(in_file=out_csv), name='qap_functional_temporal_to_csv')

#     workflow.connect(tsnr, 'tsnr_file', temporal, 'tsnr_volume')
#     workflow.connect(temporal, 'qc', temporal_to_csv, '_outputs')
#     resource_pool['qap_functional_temporal'] = (temporal_to_csv, 'csv_file')
#     return workflow, resource_pool
