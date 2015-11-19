base_test_dir = "/tdata/QAP/qc_test"

test_sub_dir = "test_data/1019436/session_1"


def test_workflow_anatomical_reorient():

    ''' unit test for the anatomical reorient workflow BUILDER '''

    import os
    import commands

    import pkg_resources as p
    from qap.workflow_utils import build_test_case


    anat_scan = p.resource_filename("qap", os.path.join(test_sub_dir, \
                                    "anat_1", \
                                    "anatomical_scan", \
                                    "mprage.nii.gz"))

    ref_graph = p.resource_filename("qap", os.path.join("test_data", \
                                    "workflow_reference", \
                                    "anatomical_reorient", \
                                    "graph_anatomical_reorient.dot"))

    ref_inputs = p.resource_filename("qap", os.path.join("test_data", \
                                     "workflow_reference", \
                                     "anatomical_reorient", \
                                     "wf_inputs.txt"))


    # build the workflow and return it
    wf, base_dir = _run_anatomical_reorient(anat_scan, False)


    # get the workflow inputs of the workflow being tested
    wf_inputs_string = str(wf.inputs).replace("\n","")

    wf_inputs_string = wf_inputs_string.replace(base_dir, \
                           "BASE_DIRECTORY_HERE")
    wf_inputs_string = wf_inputs_string.replace(anat_scan, "IN_FILE_HERE")


    flag, err = build_test_case(wf, ref_inputs, ref_graph, wf_inputs_string)


    assert flag == 2, err

def _run_flirt_anatomical_linear_registration(anatomical_brain, \
                                                 template_brain, run=True):

    # stand-alone runner for FSL FLIRT anatomical linear registration workflow

    import os
    import sys

    import glob

    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe

    workflow = pe.Workflow(name='flirt_anatomical_linear_registration_' \
                                'workflow')

    current_dir = os.getcwd()

    workflow_dir = os.path.join(current_dir, "flirt_anatomical_linear_" \
                                    "registration")
    workflow.base_dir = workflow_dir


    num_cores_per_subject = 1


    resource_pool = {}
    config = {}


    resource_pool["anatomical_brain"] = anatomical_brain
    config["template_brain_for_anat"] = template_brain

    workflow, resource_pool = \
        flirt_anatomical_linear_registration(workflow, resource_pool, config)


    ds = pe.Node(nio.DataSink(), name='datasink_flirt_anatomical_linear_' \
                                      'registration')
    ds.inputs.base_directory = workflow_dir

    node, out_file = resource_pool["flirt_linear_warped_image"]

    workflow.connect(node, out_file, ds, 'flirt_linear_warped_image')

    if run == True:

        workflow.run(plugin='MultiProc', plugin_args= \
                         {'n_procs': num_cores_per_subject})


        outpath = glob.glob(os.path.join(workflow_dir, "flirt_linear_" \
                                         "warped_image", "*"))[0]

        return outpath

    else:

        return workflow, workflow.base_dir


def test_workflow_anatomical_skullstrip():

    ''' unit test for the anatomical skullstrip workflow BUILDER '''

    import os
    import commands

    import pkg_resources as p
    from qap.workflow_utils import build_test_case


    anat_reorient = p.resource_filename("qap", os.path.join(test_sub_dir, \
                                        "anat_1", \
                                        "anatomical_reorient", \
                                        "mprage_resample.nii.gz"))

    ref_graph = p.resource_filename("qap", os.path.join("test_data", \
                                    "workflow_reference", \
                                    "anatomical_skullstrip", \
                                    "graph_anatomical_skullstrip.dot"))

    ref_inputs = p.resource_filename("qap", os.path.join("test_data", \
                                     "workflow_reference", \
                                     "anatomical_skullstrip", \
                                     "wf_inputs.txt"))


    # build the workflow and return it
    wf, base_dir = _run_anatomical_skullstrip(anat_reorient, False)


    # get the workflow inputs of the workflow being tested
    wf_inputs_string = str(wf.inputs).replace("\n","")

    wf_inputs_string = wf_inputs_string.replace(base_dir, \
                           "base_directory_here")
    wf_inputs_string = wf_inputs_string.replace(anat_reorient, "in_file_here", 1)
    wf_inputs_string = wf_inputs_string.replace(anat_reorient, "in_file_a_here")


    flag, err = build_test_case(wf, ref_inputs, ref_graph, wf_inputs_string)


    assert flag == 2, err



def test_workflow_flirt_anatomical_linear_registration():

    ''' unit test for the anatomical reorient workflow BUILDER '''

    import os
    import pkg_resources as p

    from qap.workflow_utils import build_test_case

    anat_brain = p.resource_filename("qap", os.path.join(test_sub_dir, \
                                     "anat_1", \
                                     "anatomical_brain", \
                                     "mprage_resample_calc.nii.gz"))

    template_brain = p.resource_filename("qap", os.path.join("test_data", \
                                         "MNI152_T1_2mm_brain.nii.gz"))

    ref_graph = p.resource_filename("qap", os.path.join("test_data", \
                                    "workflow_reference", \
                                    "flirt_anatomical_linear_registration", \
                                    "graph_flirt_anatomical_linear" \
                                    "_registration.dot"))

    ref_inputs = p.resource_filename("qap", os.path.join("test_data", \
                                     "workflow_reference", \
                                     "flirt_anatomical_linear_registration", \
                                     "wf_inputs.txt"))

    # build the workflow and return it
    wf, base_dir = _run_flirt_anatomical_linear_registration(anat_brain, \
                                                            template_brain, \
                                                            False)

    # get the workflow inputs of the workflow being tested
    wf_inputs_string = str(wf.inputs).replace("\n","")

    wf_inputs_string = wf_inputs_string.replace(base_dir, \
                           "base_directory_here")
    wf_inputs_string = wf_inputs_string.replace(anat_brain, "in_file_here")
    wf_inputs_string = wf_inputs_string.replace(template_brain, \
                                                    "reference_here")


    flag, err = build_test_case(wf, ref_inputs, ref_graph, wf_inputs_string)


    assert flag == 2, err



def test_workflow_segmentation():

    ''' unit test for the segmentation workflow BUILDER '''

    import os
    import commands

    import pkg_resources as p
    from qap.workflow_utils import build_test_case

    anat_brain = p.resource_filename("qap", os.path.join(test_sub_dir, \
                                     "anat_1", \
                                     "anatomical_brain", \
                                     "mprage_resample_calc.nii.gz"))

    ref_graph = p.resource_filename("qap", os.path.join("test_data", \
                                    "workflow_reference", \
                                    "segmentation", \
                                    "graph_segmentation.dot"))

    ref_inputs = p.resource_filename("qap", os.path.join("test_data", \
                                     "workflow_reference", \
                                     "segmentation", \
                                     "wf_inputs.txt"))


    # build the workflow and return it
    wf, base_dir = _run_segmentation_workflow(anat_brain, False)


    # get the workflow inputs of the workflow being tested
    wf_inputs_string = str(wf.inputs).replace("\n","")

    wf_inputs_string = wf_inputs_string.replace(base_dir, \
                           "base_directory_here")

    list_input = "['" + anat_brain + "']"

    wf_inputs_string = wf_inputs_string.replace(list_input, "in_files_here")


    flag, err = build_test_case(wf, ref_inputs, ref_graph, wf_inputs_string)


    assert flag == 2, err


def _run_anatomical_reorient(anatomical_scan, run=True):

    # stand-alone runner for anatomical reorient workflow

    import os
    import sys

    import glob

    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe

    workflow = pe.Workflow(name='anatomical_reorient_workflow')

    current_dir = os.getcwd()

    workflow_dir = os.path.join(current_dir, "anatomical_reorient")
    workflow.base_dir = workflow_dir


    resource_pool = {}
    config = {}
    num_cores_per_subject = 1


    resource_pool["anatomical_scan"] = anatomical_scan

    workflow, resource_pool = \
            anatomical_reorient_workflow(workflow, resource_pool, config)


    ds = pe.Node(nio.DataSink(), name='datasink_anatomical_reorient')
    ds.inputs.base_directory = workflow_dir

    node, out_file = resource_pool["anatomical_reorient"]

    workflow.connect(node, out_file, ds, 'anatomical_reorient')


    if run == True:

        workflow.run(plugin='MultiProc', plugin_args= \
                         {'n_procs': num_cores_per_subject})

        outpath = glob.glob(os.path.join(workflow_dir, "anatomical_reorient",\
                                         "*"))[0]

        return outpath

    else:

        return workflow, workflow.base_dir


def _run_anatomical_skullstrip(anatomical_reorient, run=True):

    # stand-alone runner for anatomical skullstrip workflow

    import os
    import sys

    import glob

    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe

    workflow = pe.Workflow(name='anatomical_skullstrip_workflow')

    current_dir = os.getcwd()

    workflow_dir = os.path.join(current_dir, "anatomical_skullstrip")
    workflow.base_dir = workflow_dir


    resource_pool = {}
    config = {}
    num_cores_per_subject = 1


    resource_pool["anatomical_reorient"] = anatomical_reorient

    workflow, resource_pool = \
            anatomical_skullstrip_workflow(workflow, resource_pool, config)


    ds = pe.Node(nio.DataSink(), name='datasink_anatomical_skullstrip')
    ds.inputs.base_directory = workflow_dir

    node, out_file = resource_pool["anatomical_brain"]

    workflow.connect(node, out_file, ds, 'anatomical_brain')

    if run == True:

        workflow.run(plugin='MultiProc', plugin_args= \
                         {'n_procs': num_cores_per_subject})

        outpath = glob.glob(os.path.join(workflow_dir, "anatomical_brain", \
                                         "*"))[0]

        return outpath

    else:

        return workflow, workflow.base_dir


def _run_ants_anatomical_linear_registration(anatomical_brain, \
                                                template_brain, num_cores=1, \
                                                run=True):

    # stand-alone runner for anatomical skullstrip workflow

    import os
    import sys

    import glob

    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe

    workflow = pe.Workflow(name='ants_anatomical_linear_registration_' \
                                'workflow')

    current_dir = os.getcwd()

    workflow_dir = os.path.join(current_dir, "ants_anatomical_linear_" \
                                    "registration")
    workflow.base_dir = workflow_dir


    resource_pool = {}
    config = {}
    num_cores_per_subject = 1


    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(num_cores)


    resource_pool["anatomical_brain"] = anatomical_brain
    config["template_brain_for_anat"] = template_brain

    workflow, resource_pool = \
          ants_anatomical_linear_registration(workflow, resource_pool, config)


    ds = pe.Node(nio.DataSink(), name='datasink_ants_anatomical_linear_' \
                                      'registration')
    ds.inputs.base_directory = workflow_dir

    node, out_file = resource_pool["ants_linear_warped_image"]

    workflow.connect(node, out_file, ds, 'ants_linear_warped_image')

    if run == True:

        workflow.run(plugin='MultiProc', plugin_args= \
                         {'n_procs': num_cores_per_subject})

        outpath = glob.glob(os.path.join(workflow_dir, "ants_linear_warped_" \
                                         "image", "*"))[0]

        return outpath

    else:

        return workflow


def _run_segmentation_workflow(anatomical_brain, run=True):

    # stand-alone runner for segmentation workflow

    import os
    import sys

    import glob

    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe

    workflow = pe.Workflow(name='segmentation_workflow')

    current_dir = os.getcwd()

    workflow_dir = os.path.join(current_dir, "segmentation")
    workflow.base_dir = workflow_dir


    resource_pool = {}
    config = {}
    num_cores_per_subject = 1


    resource_pool["anatomical_brain"] = anatomical_brain

    workflow, resource_pool = \
            segmentation_workflow(workflow, resource_pool, config)


    ds = pe.Node(nio.DataSink(), name='datasink_segmentation')
    ds.inputs.base_directory = workflow_dir


    seg_types = ["gm", "wm", "csf"]

    for seg in seg_types:

        node, out_file = resource_pool["anatomical_%s_mask" % seg]

        workflow.connect(node, out_file, ds, 'anatomical_%s_mask' % seg)


    if run == True:

        workflow.run(plugin='MultiProc', plugin_args= \
                         {'n_procs': num_cores_per_subject})

        outpath = glob.glob(os.path.join(workflow_dir, "anatomical_*_mask", \
                                         "*"))

        return outpath

    else:

        return workflow, workflow.base_dir



def run_all_tests_anatomical_preproc():

    test_workflow_anatomical_reorient()
    test_workflow_anatomical_skullstrip()
    test_workflow_flirt_anatomical_linear_registration()
    test_workflow_segmentation()


