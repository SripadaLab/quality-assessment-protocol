#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 18:40:40
# @Last Modified by:   Oscar Esteban
# @Last Modified time: 2015-11-20 10:33:38

import nipype.pipeline.engine as pe
from nipype.interfaces.base import (traits, InputMultiPath, CommandLine,
                                    Undefined, TraitedSpec, Bunch,
                                    DynamicTraitedSpec, InterfaceResult,
                                    md5, Interface, TraitDictObject,
                                    TraitListObject, isdefined)
from nipype.interfaces.utility import IdentityInterface
from nipype import logging
logger = logging.getLogger('workflow')


class ConditionalWorkflow(pe.Workflow):
    """
    Implements a kind of workflow that is executed depending on whether
    a control input is set or not."""

    def __init__(self, name, base_dir=None, condition_map=[]):
        """Create a workflow object.

        Parameters
        ----------
        name : alphanumeric string
            unique identifier for the workflow
        base_dir : string, optional
            path to workflow storage

        """

        super(ConditionalWorkflow, self).__init__(name, base_dir)

        if condition_map is None or not condition_map:
            raise ValueError('ConditionalWorkflow condition_map must be a '
                             'non-empty list of tuples')

        if isinstance(condition_map, tuple):
            condition_map = [condition_map]

        cond_in, cond_out = zip(*condition_map)

        self._condition = pe.Node(IdentityInterface(fields=cond_in),
                                  name='cond_fields')
        self.add_nodes([self._condition])
        self._map = condition_map

    @property
    def conditions(self):
        return self._condition.inputs

    def set_condition(self, parameter, val):
        setattr(self.conditions, parameter, deepcopy(val))

    def run(self, plugin=None, plugin_args=None, updatehash=False):
        node = self._condition
        condset = False
        outputs = []
        # Detect if conditional input is set
        for key, dest in self._map:
            outputs.append(dest)
            value = getattr(node.inputs, key)
            if condset and not isdefined(value):
                raise RuntimeError('One or more conditions are not set')
            condset = isdefined(value)

        # If so, remove all nodes and copy conditional input to output.
        if condset:
            logger.info('ConditionalWorkflow has conditions set, not running')
            out_ports = {}
            for key, trait in self.outputs.items():
                if 'cond_fields' not in key:
                    out_ports[key] = [
                        k for k, v in list(getattr(self.outputs, key).items())]
            self.remove_nodes(self._graph.nodes())

            nodes = {}
            out_keys = []
            for k, v in out_ports.iteritems():
                for name in v:
                    out_keys.append('%s.%s' % (k, name))
                nodes[k] = pe.Node(IdentityInterface(fields=v), name=k)

            for key, dest in self._map:
                if dest not in out_keys:
                    raise RuntimeError('Unknown output found when mapping')

                dest = dest.split('.')
                self.connect(self._condition, key, nodes[dest[0]], dest[1])

        # else, normally run
        return super(ConditionalWorkflow, self).run(plugin, plugin_args,
                                                    updatehash)

    def _get_conditions(self):
        """Returns the conditions of this workflow
        """
        node = self._condition
        condict = TraitedSpec()

        for key, trait in list(node.inputs.items()):
            condict.add_trait(key, traits.Trait(trait, node=node))
            value = getattr(node.inputs, key)
            setattr(condict, key, value)

        return condict

    def _set_condition(self, object, name, newvalue):
        object.traits()[name].node.set_input(name, newvalue)

