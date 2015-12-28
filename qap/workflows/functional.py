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


def qap_functional_spatial_workflow(workflow, config, plot_mask=False):
    import nipype.algorithms.misc as nam
    from utils import qap_functional_spatial
    from qap.viz.interfaces import PlotMosaic

    settings = ['subject_id', 'session_id', 'scan_id', 'site_name',
                'direction']
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['functional_scan', 'start_idx', 'stop_idx']+settings),
        name='inputnode')

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
    from qap.viz.interfaces import PlotMosaic

    settings = ['subject_id', 'session_id', 'scan_id', 'site_name',
                'direction']
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['functional_scan', 'start_idx', 'stop_idx']+settings),
        name='inputnode')

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

    return workflow
