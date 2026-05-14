## Install R dependencies for brms models.
## Run this once on the cluster before submitting model jobs.
## CmdStan compilation takes ~30-60 minutes.
## Safe to rerun -- skips anything already installed.

packages <- c("dplyr", "readr", "here", "brms", "cmdstanr")

to_install <- packages[!packages %in% installed.packages()[, "Package"]]
if (length(to_install) > 0) {
  install.packages(
    to_install,
    repos = c("https://stan-dev.r-universe.dev", "https://cloud.r-project.org")
  )
} else {
  message("All R packages already installed.")
}

library(cmdstanr)

# Find existing CmdStan installation or install fresh
default_dir <- cmdstan_default_install_dir()
installed_versions <- list.files(default_dir, pattern = "^cmdstan-")

if (length(installed_versions) > 0) {
  latest <- file.path(default_dir, sort(installed_versions)[length(installed_versions)])
  set_cmdstan_path(latest)
  message(paste("CmdStan path set to:", cmdstan_version()))
} else {
  message("Installing CmdStan...")
  install_cmdstan(cores = 4)
}

# Persist path in ~/.Rprofile so future sessions find CmdStan automatically
rprofile <- path.expand("~/.Rprofile")
path_line <- paste0('cmdstanr::set_cmdstan_path("', cmdstan_path(), '")')
existing <- if (file.exists(rprofile)) readLines(rprofile) else character(0)
if (!any(grepl("set_cmdstan_path", existing))) {
  cat(path_line, file = rprofile, append = TRUE, sep = "\n")
  message("Added set_cmdstan_path to ~/.Rprofile")
} else {
  message("~/.Rprofile already contains set_cmdstan_path -- no changes needed")
}

message("Done. Verify with check_cmdstan_toolchain():")
check_cmdstan_toolchain()
