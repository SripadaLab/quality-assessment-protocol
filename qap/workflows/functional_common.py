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
from nipype.interfaces.afni import preprocess as afp

from nipype import logging
logger = logging.getLogger('workflow')


def functional_brain_mask_workflow(name='QAPFunctBrainMask', use_bet=False,
                                   slice_timing_correction=False):

    wf = pe.CachedWorkflow(name=name, cache_map=(
        'functional_brain_mask', 'functional_brain_mask'))

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['functional_scan', 'start_idx', 'stop_idx']), name='inputnode')

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
            (func_get_brain_mask, 'output', [
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
            (erode_one_voxel, 'output', [
                ('out_file', 'functional_brain_mask')])
        ])

    return wf


def func_motion_correct_workflow(name='QAPFunctionalHMC',
                                 slice_timing_correction=False):
    """
    A head motion correction (HMC) workflow for functional scans
    """

    def _getidx(in_files, start_idx, stop_idx):
        from nibabel import load
        from nipype.interfaces.base import isdefined
        nvols = load(in_files).shape[3]
        max_idx = nvols - 1

        if (not isdefined(start_idx) or start_idx < 0 or start_idx > max_idx):
            start_idx = 0

        if (not isdefined(stop_idx) or stop_idx < start_idx or
                stop_idx > max_idx):
            stop_idx = max_idx
        return start_idx, stop_idx

    wf = pe.CachedWorkflow(name=name, cache_map=(
        'func_motion_correct', 'func_motion_correct'))

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['functional_scan', 'start_idx', 'stop_idx']), name='inputnode')

    func_get_idx = pe.Node(niu.Function(
        input_names=['in_files', 'start_idx', 'stop_idx'],
        output_names=['start_idx', 'stop_idx'], function=_getidx),
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
        (func_hmc_A, 'output',   [
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

    wf = pe.CachedWorkflow(name=name, cache_map=(
        'mean_functional', 'mean_functional'))

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['functional_scan', 'start_idx', 'stop_idx']), name='inputnode')

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
        (func_mean_skullstrip, 'output',   [('out_file', 'mean_functional')])
    ])
    return wf
