## Install R dependencies for brms models.
## Run this once on the cluster before submitting model jobs.
## CmdStan compilation takes ~30-60 minutes.

packages <- c("tidyverse", "here", "brms", "cmdstanr")

# Install any packages not already present
to_install <- packages[!packages %in% installed.packages()[, "Package"]]
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
} else {
  message("All R packages already installed.")
}

library(cmdstanr)

if (is.null(cmdstan_version(error_on_NA = FALSE))) {
  message("Installing CmdStan...")
  install_cmdstan(cores = 4)
} else {
  message(paste("CmdStan already installed:", cmdstan_version()))
}

message("Done. Verify with check_cmdstan_toolchain():")
check_cmdstan_toolchain()
