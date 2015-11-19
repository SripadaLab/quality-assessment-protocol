#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 18:40:40
# @Last Modified by:   oesteban
# @Last Modified time: 2015-11-19 19:05:02

import nipype.pipeline.engine as pe


class ConditionalWorkflow(pe.Workflow):
    """
    Implements a kind of workflow that is executed depending on whether
    a control input is set or not."""

    def run(self, plugin=None, plugin_args=None, updatehash=False):
        # Detect if conditional input is set
        # If so, remove all nodes and copy conditional input to output.
        # else, normally run
        return super(ConditionalWorkflow, self).run(plugin, plugin_args,
                                                    updatehash)
