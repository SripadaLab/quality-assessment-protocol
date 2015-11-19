#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os.path as op


def qap_anatomical_spatial_workflow(workflow, resource_pool, config,
                                    plot_mask=False):
    import os
    import sys

    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe
    import nipype.interfaces.utility as niu
    import nipype.algorithms.misc as nam
    from qap_workflows_utils import qap_anatomical_spatial
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
        from anatomical_preproc import anatomical_reorient_workflow
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
        from anatomical_preproc import anatomical_skullstrip_workflow
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
        from anatomical_preproc import qap_mask_workflow
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
        from anatomical_preproc import segmentation_workflow
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


def qap_functional_spatial_workflow(workflow, resource_pool, config):

    # resource pool should have:
    #     mean_functional
    #     functional_brain_mask

    import os
    import sys

    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe

    import nipype.algorithms.misc as nam
    import nipype.interfaces.utility as niu
    import nipype.algorithms.misc as nam

    from qap_workflows_utils import qap_functional_spatial
    from qap.viz.interfaces import PlotMosaic

    from workflow_utils import check_input_resources

    if 'mean_functional' not in resource_pool.keys():
        from functional_preproc import mean_functional_workflow
        workflow, resource_pool = \
            mean_functional_workflow(workflow, resource_pool, config)

    if 'functional_brain_mask' not in resource_pool.keys():
        from functional_preproc import functional_brain_mask_workflow
        workflow, resource_pool = \
            functional_brain_mask_workflow(workflow, resource_pool, config)

    spatial_epi = pe.Node(niu.Function(
        input_names=['mean_epi', 'func_brain_mask', 'direction', 'subject_id',
                     'session_id', 'scan_id', 'site_name'],
        output_names=['qc'], function=qap_functional_spatial),
        name='qap_functional_spatial')

    # Subject infos
    if 'ghost_direction' not in config.keys():
        config['ghost_direction'] = 'y'

    spatial_epi.inputs.direction = config['ghost_direction']
    spatial_epi.inputs.subject_id = config['subject_id']
    spatial_epi.inputs.session_id = config['session_id']
    spatial_epi.inputs.scan_id = config['scan_id']

    if 'site_name' in config.keys():
        spatial_epi.inputs.site_name = config['site_name']

    if len(resource_pool['mean_functional']) == 2:
        node, out_file = resource_pool['mean_functional']
        workflow.connect(node, out_file, spatial_epi, 'mean_epi')
    else:
        spatial_epi.inputs.mean_epi = resource_pool['mean_functional']

    if len(resource_pool['functional_brain_mask']) == 2:
        node, out_file = resource_pool['functional_brain_mask']
        workflow.connect(node, out_file, spatial_epi, 'func_brain_mask')
    else:
        spatial_epi.inputs.func_brain_mask = \
            resource_pool['functional_brain_mask']

    if config.get('write_report', False):
        plot = pe.Node(PlotMosaic(), name='plot_mosaic')
        plot.inputs.subject = config['subject_id']

        metadata = [config['session_id'], config['scan_id']]
        if 'site_name' in config.keys():
            metadata.append(config['site_name'])

        plot.inputs.metadata = metadata
        plot.inputs.title = 'Mean EPI'

        if len(resource_pool['mean_functional']) == 2:
            node, out_file = resource_pool['mean_functional']
            workflow.connect(node, out_file, plot, 'in_file')
        else:
            plot.inputs.in_file = resource_pool['mean_functional']

        # Enable this if we want masks
        # if len(resource_pool['functional_brain_mask']) == 2:
        #     node, out_file = resource_pool['functional_brain_mask']
        #     workflow.connect(node, out_file, plot, 'in_mask')
        # else:
        #     plot.inputs.in_mask = resource_pool['functional_brain_mask']
        resource_pool['qap_mosaic'] = (plot, 'out_file')

    out_csv = op.join(
        config['output_directory'], 'qap_functional_spatial.csv')
    spatial_epi_to_csv = pe.Node(
        nam.AddCSVRow(in_file=out_csv), name='qap_functional_spatial_to_csv')
    workflow.connect(spatial_epi, 'qc', spatial_epi_to_csv, '_outputs')
    resource_pool['qap_functional_spatial'] = (spatial_epi_to_csv, 'csv_file')

    return workflow, resource_pool


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

    from qap_workflows_utils import qap_functional_temporal
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
