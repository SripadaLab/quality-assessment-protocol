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

from workflow_utils import check_input_resources, check_config_settings


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
        fields=['anatomical_reorient', 'anatomical_brain']))
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['head_mask'], name='outputnode'))

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
    from anatomical_preproc_utils import pick_seg_type

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
            function=pick_seg_type), name='pick_%s' % seg)
        pick_seg.inputs.seg_type = seg

        wf.connect([
            (segment, pick_seg, [('tissue_class_files', 'probability_maps')]),
            (pick_seg, outputnode, [('filename', 'anatomical_%s_map' % seg)])
        ])
    return wf


def ants_anatomical_linear_registration(workflow, resource_pool, config):

    # resource pool should have:
    #     anatomical_brain

    # linear ANTS registration takes roughly 2.5 minutes per subject running
    # on one core of an Intel Core i7-4800MQ CPU @ 2.70GHz

    import os
    import sys

    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe


    import nipype.interfaces.utility as util

    from anatomical_preproc_utils import ants_lin_reg, \
                                         separate_warps_list

    from workflow_utils import check_input_resources, \
                               check_config_settings
    from nipype.interfaces.fsl.base import Info

    if "template_brain_for_anat" not in config:
        config["template_brain_for_anat"] = Info.standard_image("MNI152_T1_2mm_brain.nii.gz")
    check_config_settings(config, "template_brain_for_anat")


    if "anatomical_brain" not in resource_pool.keys():

        from anatomical_preproc import anatomical_skullstrip_workflow

        workflow, resource_pool = \
            anatomical_skullstrip_workflow(workflow, resource_pool, config)


    #check_input_resources(resource_pool, "anatomical_brain")


    calc_ants_warp = pe.Node(niu.Function(
                                 input_names=['anatomical_brain',
                                              'reference_brain'],
                                 output_names=['warp_list',
                                               'warped_image'],
                                 function=ants_lin_reg),
                                 name='calc_ants_linear_warp')


    select_forward_initial = pe.Node(niu.Function(input_names=['warp_list',
            'selection'], output_names=['selected_warp'],
            function=separate_warps_list), name='select_forward_initial')

    select_forward_initial.inputs.selection = "Initial"


    select_forward_rigid = pe.Node(niu.Function(input_names=['warp_list',
            'selection'], output_names=['selected_warp'],
            function=separate_warps_list), name='select_forward_rigid')

    select_forward_rigid.inputs.selection = "Rigid"


    select_forward_affine = pe.Node(niu.Function(input_names=['warp_list',
            'selection'], output_names=['selected_warp'],
            function=separate_warps_list), name='select_forward_affine')

    select_forward_affine.inputs.selection = "Affine"


    if len(resource_pool["anatomical_brain"]) == 2:
        node, out_file = resource_pool["anatomical_brain"]
        workflow.connect(node, out_file, calc_ants_warp, 'anatomical_brain')
    else:
       calc_ants_warp.inputs.anatomical_brain = \
            resource_pool["anatomical_brain"]


    calc_ants_warp.inputs.reference_brain = config["template_brain_for_anat"]


    workflow.connect(calc_ants_warp, 'warp_list',
                         select_forward_initial, 'warp_list')

    workflow.connect(calc_ants_warp, 'warp_list',
                         select_forward_rigid, 'warp_list')

    workflow.connect(calc_ants_warp, 'warp_list',
                         select_forward_affine, 'warp_list')


    resource_pool["ants_initial_xfm"] = \
        (select_forward_initial, 'selected_warp')

    resource_pool["ants_rigid_xfm"] = (select_forward_rigid, 'selected_warp')

    resource_pool["ants_affine_xfm"] = \
        (select_forward_affine, 'selected_warp')

    resource_pool["ants_linear_warped_image"] = \
        (calc_ants_warp, 'warped_image')

    return workflow, resource_pool
