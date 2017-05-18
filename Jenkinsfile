#!/usr/bin/env groovy

// Generated from snippet generator 'properties; set job properties'
properties([buildDiscarder(logRotator(
    artifactDaysToKeepStr: '',
    artifactNumToKeepStr: '',
    daysToKeepStr: '',
    numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

////////////////////////////////////////////////////////////////////////
// -- HELPER AUXILLARY FUNCTIONS

////////////////////////////////////////////////////////////////////////
// Construct the relative path of the build directory
String build_directory_rel( String build_config )
{
  if( build_config.equalsIgnoreCase( 'release' ) )
  {
    return "build/release"
  }
  else
  {
    return "build/debug"
  }
}

////////////////////////////////////////////////////////////////////////
// -- FUNCTIONS RELATED TO BUILD
// This encapsulates running of unit tests
def docker_build_image( String src_dir_abs )
{
  String project = "hipeigen"
  String build_type_name = "build-ubuntu-16.04"
  String dockerfile_name = "dockerfile-${build_type_name}"
  String build_image_name = "${build_type_name}"
  def build_image = null

  stage('ubuntu-16.04 image')
  {
    dir("${src_dir_abs}/hipeigen/docker")
    {
      def user_uid = sh( script: 'id -u', returnStdout: true ).trim()
      build_image = docker.build( "${project}/${build_image_name}:latest", "-f ${dockerfile_name} --build-arg user_uid=${user_uid} ." )
    }
  }

  return build_image
}

////////////////////////////////////////////////////////////////////////
// Checkout the desired source code and update the version number
String checkout_and_version( String workspace_dir_abs )
{
  String source_dir_abs = "${workspace_dir_abs}/src/"

  stage("github clone")
  {
    dir( "${source_dir_abs}/hipeigen" )
    {
      deleteDir( )

      // checkout hipeigen
      checkout([
          $class: 'GitSCM',
          branches: scm.branches,
          doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
          extensions: scm.extensions + [[$class: 'CloneOption', depth: 1, noTags: false, reference: '', shallow: true]],
          submoduleCfg: [],
          userRemoteConfigs: scm.userRemoteConfigs
          ])
    }
  }

  return source_dir_abs
}


////////////////////////////////////////////////////////////////////////
// This encapsulates the cmake configure, build and package commands
// Leverages docker containers to encapsulate the build in a fixed environment
def docker_build_inside_image( def build_image, String build_config, String src_dir_abs, String build_dir_rel )
{
  build_image.inside( '--device=/dev/kfd' )
  {
    // withEnv(["HCC_AMDGPU_TARGET=gfx900", "HIPCC_VERBOSE=0"])
    // {
      stage("make ${build_config}")
      {
        sh  """#!/usr/bin/env bash
            set -x
            /opt/rocm/bin/hipconfig
            printenv | sort
            cd ${build_dir_rel}
            rm -rf *
            cmake ${src_dir_abs}/hipeigen
            sudo make -j \$(nproc) install
          """
      }

      stage("make test")
      {
        sh  """#!/usr/bin/env bash
            set -x
            cd ${build_dir_rel}
            make -j \$(nproc) buildtests
            ctest --no-compress-output --output-on-failure -T test || true
          """

        step([$class: 'XUnitBuilder', testTimeMargin: '3000', thresholdMode: 1,
          thresholds:
            [[$class: 'FailedThreshold', failureNewThreshold: '', failureThreshold: '0', unstableNewThreshold: '', unstableThreshold: ''],
            [$class: 'SkippedThreshold', failureNewThreshold: '', failureThreshold: '', unstableNewThreshold: '', unstableThreshold: '']],
          tools:
            [[$class: 'CTestType', deleteOutputFiles: true, failIfNotNew: true, pattern: "${build_dir_rel}/Testing/**/*.xml", skipNoTestFiles: false, stopProcessingIfError: true]]])
      }
    // }
  }

  return void
}

////////////////////////////////////////////////////////////////////////
// This routines defines the pipeline of the build; the order that various helper functions
// are called.
// Calls helper routines to do the work and stitches them together
def hipeigen_build_pipeline( String build_config )
{
  // Convenience variables for common paths used in building
  String workspace_dir_abs = pwd()

  // Checkout all dependencies
  String source_dir_abs = checkout_and_version( "${workspace_dir_abs}" )

  // Create/reuse a docker image that represents the hipeigen build environment
  def hipeigen_build_image = docker_build_image( "${source_dir_abs}" )

  String build_dir_rel = build_directory_rel( build_config );

  // Build hipeigen inside of the build environment
  docker_build_inside_image( hipeigen_build_image, "${build_config}", "${source_dir_abs}", "${build_dir_rel}" )

  return void
}

////////////////////////////////////////////////////////////////////////
// -- MAIN
// This following are build nodes; start of build pipeline
node('docker && rocm && gfx803')
{
  hipeigen_build_pipeline( 'Release' )
}
