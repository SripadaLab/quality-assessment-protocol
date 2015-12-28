#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os.path as op
import workflows.wrappers as qapw


def qap_anatomical_spatial_workflow(workflow, resource_pool, config,
                                    plot_mask=False):
    return qapw.qap_anatomical_spatial_workflow(
        workflow, resource_pool, config, plot_mask)


def qap_functional_spatial_workflow(workflow, resource_pool, config,
                                    plot_mask=False):
    return qapw.qap_functional_spatial_workflow(
        workflow, resource_pool, config, plot_mask)


def qap_functional_temporal_workflow(workflow, resource_pool, config,
                                     plot_mask=False):
    return qapw.qap_functional_temporal_workflow(
        workflow, resource_pool, config, plot_mask)
