#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import sys

import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.interfaces.fsl.maths as fsl

from qap.workflow_utils import check_input_resources, check_config_settings


def qap_anatomical_spatial_workflow(workflow, resource_pool, config,
                                    plot_mask=False):
    import nipype.algorithms.misc as nam
    from qap.qap_workflows_utils import qap_anatomical_spatial
    from qap.viz.interfaces import PlotMosaic

    # resource pool should have:
    inputs = ['anatomical_reorient', 'head_mask_path', 'anatomical_gm_mask',
              'anatomical_wm_mask', 'anatomical_csf_mask']

    inputnode = pe.Node(niu.IdentityInterface(
        fields=inputs + ['anatomical_scan', 'anatomical_brain']),
        name='inputnode')

    spatial = pe.Node(niu.Function(
        input_names=inputs + ['subject_id', 'session_id', 'scan_id',
                              'site_name'],
        output_names=['qc'], function=qap_anatomical_spatial),
        name='qap_anatomical_spatial')

    # Subject infos
    spatial.inputs.subject_id = config['subject_id']
    spatial.inputs.session_id = config['session_id']
    spatial.inputs.scan_id = config['scan_id']
    if 'site_name' in config.keys():
        spatial.inputs.site_name = config['site_name']

    # Connect images
    workflow.connect([(inputnode, spatial, [(f, f) for f in inputs])])

    # Check reoriented image, compute if not found
    if 'anatomical_reorient' not in resource_pool.keys():
        arw = anatomical_reorient_workflow()

        workflow.connect([
            (inputnode, arw,
                [('anatomical_scan', 'inputnode.anatomical_scan')]),
            (arw, inputnode,
                [('outputnode.anatomical_reorient', 'anatomical_reorient')])
        ])
    else:
        inputnode.inputs.anatomical_reorient = resource_pool[
            'anatomical_reorient']

    # Check brain image, compute if not found
    if 'anatomical_brain' not in resource_pool.keys():
        asw = anatomical_skullstrip_workflow()

        workflow.connect([
            (inputnode, asw,
                [('anatomical_reorient', 'inputnode.anatomical_reorient')]),
            (asw, inputnode,
                [('outputnode.anatomical_brain', 'anatomical_brain')])
        ])
    else:
        inputnode.inputs.anatomical_brain = resource_pool['anatomical_brain']

    # Check brain mask, compute if not found
    if 'qap_head_mask' not in resource_pool.keys():
        qmw = qap_mask_workflow(config=config)

        workflow.connect([
            (inputnode, qmw,
                [('anatomical_reorient', 'inputnode.anatomical_reorient'),
                 ('anatomical_brain', 'inputnode.anatomical_brain')]),
            (qmw, inputnode,
                [('outputnode.head_mask', 'head_mask_path')])
        ])

        node, out_file = resource_pool['qap_head_mask']
        workflow.connect(node, out_file, inputnode, 'head_mask_path')
    else:
        inputnode.inputs.head_mask_path = resource_pool['qap_head_mask']

    # Check segmentations
    if (('anatomical_gm_mask' not in resource_pool.keys()) or
        ('anatomical_wm_mask' not in resource_pool.keys()) or
            ('anatomical_csf_mask' not in resource_pool.keys())):
        qsw = segmentation_workflow()

        workflow.connect([
            (inputnode, qsw,
                [('anatomical_brain', 'inputnode.anatomical_brain')]),
            (qsw, inputnode,
                [('outputnode.anatomical_gm_mask', 'anatomical_gm_mask'),
                 ('outputnode.anatomical_wm_mask', 'anatomical_wm_mask'),
                 ('outputnode.anatomical_csf_mask', 'anatomical_csf_mask')])
        ])
    else:
        inputnode.inputs.anatomical_gm_mask = \
            resource_pool['anatomical_gm_mask']
        inputnode.inputs.anatomical_wm_mask = \
            resource_pool['anatomical_wm_mask']
        inputnode.inputs.anatomical_csf_mask = \
            resource_pool['anatomical_csf_mask']

    # Write CSV row
    out_csv = op.join(config['output_directory'], 'qap_anatomical_spatial.csv')
    spatial_to_csv = pe.Node(
        nam.AddCSVRow(in_file=out_csv), name='qap_anatomical_spatial_to_csv')
    workflow.connect(spatial, 'qc', spatial_to_csv, '_outputs')
    resource_pool['qap_anatomical_spatial'] = (spatial_to_csv, 'csv_file')

    # Append plot generation
    if config.get('write_report', False):
        plot = pe.Node(PlotMosaic(), name='plot_mosaic')
        plot.inputs.subject = config['subject_id']

        metadata = [config['session_id'], config['scan_id']]
        if 'site_name' in config.keys():
            metadata.append(config['site_name'])

        plot.inputs.metadata = metadata
        plot.inputs.title = 'Anatomical reoriented'
        workflow.connect(inputnode, 'anatomical_reorient', plot, 'in_file')
        if plot_mask:
            workflow.connect(inputnode, 'head_mask_path', plot, 'in_mask')

        resource_pool['qap_mosaic'] = (plot, 'out_file')

    return workflow, resource_pool


def anatomical_reorient_workflow(name='QAPAnatReorient'):
    """
    A workflow to reorient images to 'RPI' orientation
    """
    from nipype.interfaces.afni import preprocess as afp

    wf = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['anatomical_scan']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['anatomical_reorient']),
                         name='outputnode')
    anat_deoblique = pe.Node(afp.Refit(deoblique=True), name='anat_deoblique')
    anat_reorient = pe.Node(afp.Resample(
        orientation='RPI', outputtype='NIFTI_GZ'), name='anat_reorient')
    wf.connect([
        (inputnode, anat_deoblique,     [('anatomical_scan', 'in_file')]),
        (anat_deoblique, anat_reorient, [('out_file', 'in_file')]),
        (anat_reorient, outputnode,     [('out_file', 'anatomical_reorient')])
    ])
    return wf


def qap_mask_workflow(name='QAPMaskWorkflow', config={}):
    from nipype.interfaces.fsl.base import Info
    from qap_workflows_utils import select_thresh, slice_head_mask

    wf = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['anatomical_reorient', 'anatomical_brain']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['head_mask']), name='outputnode')

    select_thresh = pe.Node(niu.Function(
        input_names=['input_skull'], output_names=['thresh_out'],
        function=select_thresh), name='qap_headmask_select_thresh',
        iterfield=['input_skull'])

    mask_skull = pe.Node(
        fsl.Threshold(args='-bin'), name='qap_headmask_thresh')

    dilate_node = pe.Node(
        fsl.MathsCommand(args='-dilM -dilM -dilM -dilM -dilM -dilM'),
        name='qap_headmask_dilate')

    erode_node = pe.Node(
        fsl.MathsCommand(args='-eroF -eroF -eroF -eroF -eroF -eroF'),
        name='qap_headmask_erode')

    slice_head_mask = pe.Node(niu.Function(
        input_names=['infile', 'transform', 'standard'],
        output_names=['outfile_path'], function=slice_head_mask),
        name='qap_headmask_slice_head_mask')

    slice_head_mask.inputs.standard = config.get(
        'template_skull_for_anat', Info.standard_image('MNI152_T1_2mm.nii.gz'))

    combine_masks = pe.Node(fsl.BinaryMaths(
        operation='add', args='-bin'), name='qap_headmask_combine_masks')

    falrw = flirt_anatomical_linear_registration(config=config)

    wf.connect([
        (inputnode, select_thresh,   [('anatomical_reorient', 'input_skull')]),
        (inputnode, mask_skull,      [('anatomical_reorient', 'in_file')]),
        (inputnode, slice_head_mask, [('anatomical_reorient', 'infile')]),
        (inputnode, falrw,
            [('anatomical_brain', 'inputnode.anatomical_brain')]),
        (falrw, slice_head_mask,
            [('outputnode.flirt_affine_xfm', 'transform')]),
        (select_thresh, mask_skull,      [('thresh_out', 'thresh')]),
        (mask_skull, dilate_node,        [('out_file', 'in_file')]),
        (dilate_node, erode_node,        [('out_file', 'in_file')]),
        (erode_node, combine_masks,      [('out_file', 'in_file')]),
        (slice_head_mask, combine_masks, [('outfile_path', 'operand_file')]),
        (combine_masks, outputnode,      [('out_file', 'head_mask')])
    ])
    return wf


def flirt_anatomical_linear_registration(name='QAPflirtLReg', config={}):
    from nipype.interfaces.fsl.base import Info

    wf = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['anatomical_brain']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['flirt_affine_xfm', 'flirt_linear_warped_image']),
        name='outputnode')

    calc_flirt_warp = pe.Node(fsl.FLIRT(cost='corratio'),
                              name='calc_flirt_warp')

    calc_flirt_warp.inputs.reference = config.get(
        'template_brain_for_anat', Info.standard_image(
            'MNI152_T1_2mm_brain.nii.gz'))

    wf.connect([
        (inputnode, calc_flirt_warp, [('anatomical_brain', 'in_file')]),
        (calc_flirt_warp, outputnode, [
            ('out_matrix_file', 'flirt_affine_xfm'),
            ('out_file', 'flirt_linear_warped_image')]),
    ])

    return wf


def anatomical_skullstrip_workflow(name='QAPSkullStripWorkflow'):
    from nipype.interfaces.afni import preprocess as afp

    wf = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['anatomical_reorient']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['anatomical_brain']),
                         name='outputnode')
    sstrip = pe.Node(afp.SkullStrip(), name='anat_skullstrip')
    sstrip.inputs.outputtype = 'NIFTI_GZ'

    sstrip_orig_vol = pe.Node(afp.Calc(
        expr='a*step(b)', outputtype='NIFTI_GZ'), name='anat_sstrip_orig_vol')

    wf.connect([
        (inputnode, sstrip,           [('anatomical_reorient', 'in_file')]),
        (inputnode, sstrip_orig_vol,  [('anatomical_reorient', 'in_file_a')]),
        (sstrip, sstrip_orig_vol,     [('out_file', 'in_file_b')]),
        (sstrip_orig_vol, outputnode, [('out_file', 'anatomical_brain')])
    ])
    return wf


def segmentation_workflow(name='QAPSegmentationWorkflow'):
    wf = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['anatomical_brain']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['anatomical_gm_mask', 'anatomical_wm_mask',
                'anatomical_csf_mask']), name='outputnode')

    segment = pe.Node(fsl.FAST(
        img_type=1, segments=True, probability_maps=True,
        out_basename='segment'), name='segmentation')

    wf.connect(inputnode, 'anatomical_brain', segment, 'in_file')
    for seg in ["gm", "wm", "csf"]:
        pick_seg = pe.Node(niu.Function(
            input_names=['probability_maps', 'seg_type'],
            output_names=['filename'],
            function=_pick_seg_type), name='pick_%s' % seg)
        pick_seg.inputs.seg_type = seg

        wf.connect([
            (segment, pick_seg, [('tissue_class_files', 'probability_maps')]),
            (pick_seg, outputnode, [('filename', 'anatomical_%s_map' % seg)])
        ])
    return wf


def _pick_seg_type(probability_maps, seg_type):
    """
    Returns the selected probability map from the list of segmented
    probability maps

    Parameters
    ----------

    probability_maps : list (string)
        List of Probability Maps

    Returns
    -------

    file : string
        Path to segment_prob_0.nii.gz is returned

    """

    import os
    import sys

    if(isinstance(probability_maps, list)):
        if(len(probability_maps) == 1):
            probability_maps = probability_maps[0]
        for filename in probability_maps:
            if seg_type == "csf":
                if filename.endswith("_0.nii.gz"):
                    return filename
            elif seg_type == "gm":
                if filename.endswith("_1.nii.gz"):
                    return filename
            elif seg_type == "wm":
                if filename.endswith("_2.nii.gz"):
                    return filename
    return None

# DEPRECATED / UNFINISHED code ---------------------------------------
# def ants_anatomical_linear_registration(workflow, resource_pool, config):
#
#    # resource pool should have:
#    #     anatomical_brain
#
#    # linear ANTS registration takes roughly 2.5 minutes per subject running
#    # on one core of an Intel Core i7-4800MQ CPU @ 2.70GHz
#
#    import os
#    import sys
#
#    import nipype.interfaces.io as nio
#    import nipype.pipeline.engine as pe
#
#
#    import nipype.interfaces.utility as util
#
#    from anatomical_preproc_utils import ants_lin_reg, \
#                                         separate_warps_list
#
#    from workflow_utils import check_input_resources, \
#                               check_config_settings
#    from nipype.interfaces.fsl.base import Info
#
#    if "template_brain_for_anat" not in config:
#        config["template_brain_for_anat"] = Info.standard_image("MNI152_T1_2mm_brain.nii.gz")
#    check_config_settings(config, "template_brain_for_anat")
#
#
#    if "anatomical_brain" not in resource_pool.keys():
#
#        from anatomical_preproc import anatomical_skullstrip_workflow
#
#        workflow, resource_pool = \
#            anatomical_skullstrip_workflow(workflow, resource_pool, config)
#
#
#    #check_input_resources(resource_pool, "anatomical_brain")
#
#
#    calc_ants_warp = pe.Node(niu.Function(
#                                 input_names=['anatomical_brain',
#                                              'reference_brain'],
#                                 output_names=['warp_list',
#                                               'warped_image'],
#                                 function=ants_lin_reg),
#                                 name='calc_ants_linear_warp')
#
#
#    select_forward_initial = pe.Node(niu.Function(input_names=['warp_list',
#            'selection'], output_names=['selected_warp'],
#            function=separate_warps_list), name='select_forward_initial')
#
#    select_forward_initial.inputs.selection = "Initial"
#
#
#    select_forward_rigid = pe.Node(niu.Function(input_names=['warp_list',
#            'selection'], output_names=['selected_warp'],
#            function=separate_warps_list), name='select_forward_rigid')
#
#    select_forward_rigid.inputs.selection = "Rigid"
#
#
#    select_forward_affine = pe.Node(niu.Function(input_names=['warp_list',
#            'selection'], output_names=['selected_warp'],
#            function=separate_warps_list), name='select_forward_affine')
#
#    select_forward_affine.inputs.selection = "Affine"
#
#
#    if len(resource_pool["anatomical_brain"]) == 2:
#        node, out_file = resource_pool["anatomical_brain"]
#        workflow.connect(node, out_file, calc_ants_warp, 'anatomical_brain')
#    else:
#       calc_ants_warp.inputs.anatomical_brain = \
#            resource_pool["anatomical_brain"]
#
#
#    calc_ants_warp.inputs.reference_brain = config["template_brain_for_anat"]
#
#
#    workflow.connect(calc_ants_warp, 'warp_list',
#                         select_forward_initial, 'warp_list')
#
#    workflow.connect(calc_ants_warp, 'warp_list',
#                         select_forward_rigid, 'warp_list')
#
#    workflow.connect(calc_ants_warp, 'warp_list',
#                         select_forward_affine, 'warp_list')
#
#
#    resource_pool["ants_initial_xfm"] = \
#        (select_forward_initial, 'selected_warp')
#
#    resource_pool["ants_rigid_xfm"] = (select_forward_rigid, 'selected_warp')
#
#    resource_pool["ants_affine_xfm"] = \
#        (select_forward_affine, 'selected_warp')
#
#    resource_pool["ants_linear_warped_image"] = \
#        (calc_ants_warp, 'warped_image')
#
#    return workflow, resource_pool
