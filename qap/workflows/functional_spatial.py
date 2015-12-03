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
            ('mean_functional', 'conditions.mean_functional'),
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
            ('functional_brain_mask', 'conditions.functional_brain_mask')]),
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


def func_motion_correct_workflow(name='QAPFunctionalHMC',
                                 slice_timing_correction=False):
    """
    A head motion correction (HMC) workflow for functional scans
    """
    from nipype.interfaces.afni import preprocess as afp
    from utils import get_idx

    wf = pe.ConditionalWorkflow(name=name, condition_map=(
        'func_motion_correct', 'outputnode.func_motion_correct'))

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['functional_scan', 'start_idx', 'stop_idx']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['func_motion_correct', 'coordinate_transformation']),
        name='outputnode')

    func_get_idx = pe.Node(niu.Function(
        input_names=['in_files', 'stop_idx', 'start_idx'],
        output_names=['stop_idx', 'start_idx'], function=get_idx),
        name='func_get_idx')

    func_drop_trs = pe.Node(afp.Calc(expr='a', outputtype='NIFTI_GZ'),
                            name='func_drop_trs')
    func_deoblique = pe.Node(afp.Refit(deoblique=True),
                             name='func_deoblique')
    func_reorient = pe.Node(afp.Resample(
        orientation='RPI', outputtype='NIFTI_GZ'), name='func_reorient')
    func_get_mean_RPI = pe.Node(afp.TStat(
        options='-mean', outputtype='NIFTI_GZ'), name='func_get_mean_RPI')

    # calculate hmc parameters
    func_hmc = pe.Node(
        afp.Volreg(args='-Fourier -twopass', zpad=4, outputtype='NIFTI_GZ'),
        name='func_motion_correct')

    func_get_mean_motion = func_get_mean_RPI.clone('func_get_mean_motion')
    func_hmc_A = func_hmc.clone('func_motion_correct_A')
    func_hmc_A.inputs.md1d_file = 'max_displacement.1D'

    wf.connect([
        (inputnode, func_get_idx, [('start_idx', 'start_idx'),
                                   ('stop_idx', 'stop_idx'),
                                   ('functional_scan', 'in_files')]),
        (inputnode, func_drop_trs, [('functional_scan', 'in_file_a')]),
        (func_get_idx, func_drop_trs, [('start_idx', 'start_idx'),
                                       ('stop_idx', 'stop_idx')]),
        (func_deoblique, func_reorient, [('out_file', 'in_file')]),
        (func_reorient, func_get_mean_RPI, [('out_file', 'in_file')]),
        (func_reorient, func_hmc, [('out_file', 'in_file')]),
        (func_get_mean_RPI, func_hmc, [('out_file', 'basefile')]),
        (func_hmc, func_get_mean_motion, [('out_file', 'in_file')]),
        (func_reorient, func_hmc_A, [('out_file', 'in_file')]),
        (func_get_mean_motion, func_hmc_A, [('out_file', 'basefile')]),
        (func_hmc_A, outputnode, [
            ('out_file', 'func_motion_correct'),
            ('oned_matrix_save', 'coordinate_transformation')])
    ])

    if slice_timing_correction:
        st_corr = pe.Node(afp.TShift(
            outputtype='NIFTI_GZ'), name='func_slice_time_correction')
        wf.connect([
            (func_drop_trs, st_corr, [('out_file', 'in_file')]),
            (st_corr, func_deoblique, [('out_file', 'in_file')])
        ])
    else:
        wf.connect([
            (func_drop_trs, func_deoblique, [('out_file', 'in_file')])
        ])

    return wf


def mean_functional_workflow(name='QAPMeanFunctional',
                             slice_timing_correction=False):
    ''' this version does NOT remove background noise '''
    from nipype.interfaces.afni import preprocess as afp

    wf = pe.ConditionalWorkflow(name=name, condition_map=(
        'mean_functional', 'outputnode.mean_functional'))

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['functional_scan', 'start_idx', 'stop_idx']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['mean_functional']), name='outputnode')

    cachenode = pe.Node(niu.IdentityInterface(
        fields=['func_motion_correct']), name='cachenode')

    # This workflow is run twice
    # https://github.com/preprocessed-connectomes-project/quality-assessment-protocol/issues/10)
    hmcwf = func_motion_correct_workflow(
        slice_timing_correction=slice_timing_correction)
    wf.connect([
        (inputnode, hmcwf, [
            ('functional_scan', 'inputnode.functional_scan'),
            ('start_idx', 'inputnode.start_idx'),
            ('stop_idx', 'inputnode.stop_idx')]),
        (cachenode, hmcwf, [(
            ('func_motion_correct', 'conditions.func_motion_correct'))])
    ])

    func_mean_skullstrip = pe.Node(afp.TStat(
        options='-mean', outputtype='NIFTI_GZ'), name='func_mean_skullstrip')

    wf.connect([
        (hmcwf, func_mean_skullstrip, [
            ('outputnode.func_motion_correct', 'in_file')]),
        (func_mean_skullstrip, outputnode, [('out_file', 'mean_functional')])
    ])
    return wf


def functional_brain_mask_workflow(name='QAPFunctBrainMask', use_bet=False,
                                   slice_timing_correction=False):
    from nipype.interfaces.afni import preprocess as afp

    wf = pe.ConditionalWorkflow(name=name, condition_map=(
        'functional_brain_mask', 'outputnode.functional_brain_mask'))

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['functional_scan', 'start_idx', 'stop_idx']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['functional_brain_mask']), name='outputnode')

    cachenode = pe.Node(niu.IdentityInterface(
        fields=['func_motion_correct']), name='cachenode')

    # This workflow is run twice
    # https://github.com/preprocessed-connectomes-project/quality-assessment-protocol/issues/10)
    hmcwf = func_motion_correct_workflow(
        slice_timing_correction=slice_timing_correction)
    wf.connect([
        (inputnode, hmcwf, [
            ('functional_scan', 'inputnode.functional_scan'),
            ('start_idx', 'inputnode.start_idx'),
            ('stop_idx', 'inputnode.stop_idx')]),
        (cachenode, hmcwf, [(
            ('func_motion_correct', 'conditions.func_motion_correct'))])
    ])

    if not use_bet:
        func_get_brain_mask = pe.Node(afp.Automask(
            outputtype='NIFTI_GZ'), name='func_get_brain_mask')

        # Connect brain mask extraction
        wf.connect([
            (hmcwf, func_get_brain_mask, [
                ('outputnode.func_motion_correct', 'in_file')]),
            (func_get_brain_mask, outputnode, [
                ('out_file', 'functional_brain_mask')])
        ])

    else:
        from nipype.interfaces.fsl import BET, ErodeImage
        func_get_brain_mask = pe.Node(BET(
            mask=True, functional=True), name='func_get_brain_mask_BET')
        erode_one_voxel = pe.Node(ErodeImage(
            kernel_shape='box', kernel_size=1.0), name='erode_one_voxel')

        # Connect brain mask extraction
        wf.connect([
            (hmcwf, func_get_brain_mask, [
                ('outputnode.func_motion_correct', 'in_file')]),
            (func_get_brain_mask, erode_one_voxel, [('mask_file', 'in_file')]),
            (erode_one_voxel, outputnode, [
                ('out_file', 'functional_brain_mask')])
        ])

    return wf
