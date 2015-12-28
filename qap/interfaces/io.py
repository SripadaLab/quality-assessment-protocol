#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:02:17
# @Last Modified by:   oesteban
# @Last Modified time: 2015-12-03 18:35:20

import yaml
from nipype.interfaces.base import (BaseInterface, traits, TraitedSpec, File,
                                    InputMultiPath, OutputMultiPath,
                                    BaseInterfaceInputSpec, isdefined,
                                    DynamicTraitedSpec, Undefined)
from nipype.interfaces.io import IOBase
from nipype import logging

iflogger = logging.getLogger('interface')


class YAMLDataSourceInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, desc='QAP subjects YAML file')
    fields = traits.List(traits.Str(), mandatory=True,
                         desc='Search for these fields')
    subject_id = traits.Str(desc='subject identifier')
    session_id = traits.Str(desc='session identifier')
    scan_id = traits.Str(desc='scan identifier')
    flatten = traits.Bool(False, usedefault=True,
                          desc='flatten outputs')


class YAMLDataSource(IOBase):
    input_spec = YAMLDataSourceInputSpec
    output_spec = DynamicTraitedSpec
    _always_run = True

    def _list_outputs(self):
        from itertools import product

        if not isdefined(self.inputs.fields or not self.inputs.fields):
            raise RuntimeError('Please define the output fields')

        outputs = {}
        fields = self.inputs.fields
        for f in fields:
            outputs[f] = []

        with open(self.inputs.in_file, "r") as f:
            db = yaml.load(f)

        subkeys = sorted(db.keys())
        if isdefined(self.inputs.subject_id):
            if self.inputs.subject_id in subkeys:
                subkeys = [self.inputs.subject_id]
            else:
                raise IOError('Subject %s not found in yml file' %
                              self.inputs.subject_id)

        # Grab subjects
        for s in subkeys:
            subdict = db[s]
            seskeys = sorted(subdict.keys())

            if isdefined(self.inputs.session_id) and \
                    self.inputs.session_id in seskeys:
                seskeys = [self.inputs.session_id]

            # Grab sessions
            scankeys = set([])
            for ss in seskeys:
                sesdict = subdict[ss]
                fkeys = sesdict.keys()

                if isdefined(self.inputs.scan_id):
                    scankeys.add(self.inputs.scan_id)
                else:
                    for f in fields:
                        if f in fkeys:
                            for sck in sesdict[f].keys():
                                scankeys.add(sck)

        combs = product(subkeys, seskeys, sorted(list(scankeys)))

        outputs.update({'subject_id': [], 'session_id': [], 'scan_id': []})
        for subid, sesid, scnid in combs:
            outputs['subject_id'].append(subid)
            outputs['session_id'].append(sesid)
            outputs['scan_id'].append(scnid)

            for f in fields:
                try:
                    fname = op.abspath(db[subid][sesid][f][scnid])
                except KeyError:
                    fname = Undefined

                if not op.isfile(fname):
                    fname = Undefined

                outputs[f].append(fname)

        if (self.inputs.flatten and
                all([len(outputs[f]) == 1 for f in fields])):
            for k in outputs.keys():
                outputs[k] = outputs[k][0]

        return outputs
