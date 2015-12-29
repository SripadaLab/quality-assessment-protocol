#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:02:17
# @Last Modified by:   oesteban
# @Last Modified time: 2015-12-03 18:35:20

import yaml
import os.path as op
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
        if not isdefined(self.inputs.fields or not self.inputs.fields):
            raise RuntimeError('Please define the output fields')

        kwargs = {'flatten': self.inputs.flatten}
        for f in ['subject_id', 'session_id', 'scan_id']:
            kwargs[f] = None
            val = getattr(self.inputs, f, None)
            if isdefined(val):
                kwargs[f] = val

        return read_yml(self.inputs.in_file, self.inputs.fields, **kwargs)


def read_yml(in_file, fields, subject_id=None, session_id=None,
             scan_id=None, flatten=False):
    from itertools import product
    outputs = {}
    for f in fields:
        outputs[f] = []

    with open(in_file, "r") as f:
        db = yaml.load(f)

    subkeys = sorted(db.keys())
    if subject_id is not None:
        if subject_id in subkeys:
            subkeys = [subject_id]
        else:
            raise IOError('Subject %s not found in yml file' %
                          subject_id)

    # Grab subjects
    for s in subkeys:
        subdict = db[s]
        seskeys = sorted(subdict.keys())

        if session_id is not None and \
                session_id in seskeys:
            seskeys = [session_id]

        # Grab sessions
        scankeys = set([])
        for ss in seskeys:
            sesdict = subdict[ss]
            fkeys = sesdict.keys()

            if scan_id is not None:
                scankeys.add(scan_id)
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

                if not op.isfile(fname):
                    fname = Undefined
            except KeyError:
                fname = Undefined

            outputs[f].append(fname)

    n_results = len(outputs['subject_id'])

    remove_idx = []
    for i in range(n_results):
        row = [isdefined(outputs[f][i]) for f in fields]
        if not any(row):
            remove_idx.append(i)

    for i in reversed(remove_idx):
        for f in fields:
            outputs[f].pop(i)

    n_results = len(outputs['subject_id'])

    if flatten and (n_results == 1):
        for k in outputs.keys():
            outputs[k] = outputs[k][0]
    outputs['n_results'] = n_results

    return outputs
