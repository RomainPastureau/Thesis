# GOAL: Test discriminability of the speaker's mask
# Missing responses are excluded from the calculation

######################################################################
# Clear workspace
######################################################################

rm(list = ls())


######################################################################
# Libraries
######################################################################

library(readxl)
library(tidyverse)
library(psych)


######################################################################
# Paths and data
######################################################################

setwd("")

df <- read_xlsx(
  "Results_pretest.xlsx",
  na = c("", "NA", "N/A")
)

# Remove unwanted rows and columns
df <- df[-(22:24), -(2:15)]

# Remove first row
df <- df[-1, ]


######################################################################
# Reshape data
######################################################################

PilotStudy <- df %>%
  pivot_longer(
    cols = -ID,
    names_to = "video",
    values_to = "Response"
  )


######################################################################
# Add experimental variables
#
# IMPORTANT:
# This assumes that the video columns are ordered as:
# 6 masked + 6 unmasked trials for each speaker,
# in the speaker order specified below.
######################################################################

n_participants <- n_distinct(PilotStudy$ID)

speaker_order <- c(
  "K01",
  "K02",
  "K03",
  "K04",
  "K05",
  "K06"
)

trial_information <- tibble(
  video = unique(PilotStudy$video),
  Speaker = rep(speaker_order, each = 12),
  Mask = rep(
    rep(c("Masked", "Unmasked"), each = 6),
    times = length(speaker_order)
  )
)

PilotStudy <- PilotStudy %>%
  left_join(trial_information, by = "video") %>%
  mutate(
    Response = as.character(Response),
    
    Correct_Response = case_when(
      is.na(Response) ~ NA_character_,
      Mask == "Masked"   & Response == "1" ~ "correct",
      Mask == "Masked"   & Response == "0" ~ "incorrect",
      Mask == "Unmasked" & Response == "0" ~ "correct",
      Mask == "Unmasked" & Response == "1" ~ "incorrect",
      TRUE ~ NA_character_
    )
  )


######################################################################
# Check data coding
######################################################################

table(
  PilotStudy$Mask,
  PilotStudy$Response,
  PilotStudy$Correct_Response,
  useNA = "ifany"
)


######################################################################
# Compute hit rate, false-alarm rate, and d-prime
#
# Signal-present trials: Masked
# Signal-absent trials: Unmasked
#
# Hit:
# Participant responds "masked" on a masked trial.
#
# False alarm:
# Participant responds "masked" on an unmasked trial.
#
# Loglinear correction:
# rate = (count + 0.5) / (number of valid trials + 1)
######################################################################

aggr <- PilotStudy %>%
  group_by(ID) %>%
  summarise(
    n_masked = sum(Mask == "Masked" & !is.na(Response)),
    n_unmasked = sum(Mask == "Unmasked" & !is.na(Response)),
    
    hits = sum(
      Mask == "Masked" & Response == "1",
      na.rm = TRUE
    ),
    
    false_alarms = sum(
      Mask == "Unmasked" & Response == "1",
      na.rm = TRUE
    ),
    
    masked_correct = sum(
      Mask == "Masked" & Correct_Response == "correct",
      na.rm = TRUE
    ),
    
    unmasked_correct = sum(
      Mask == "Unmasked" & Correct_Response == "correct",
      na.rm = TRUE
    ),
    
    .groups = "drop"
  ) %>%
  mutate(
    hit_rate = (hits + 0.5) / (n_masked + 1),
    fa_rate = (false_alarms + 0.5) / (n_unmasked + 1),
    
    dprime = qnorm(hit_rate) - qnorm(fa_rate)
  )


######################################################################
# Inspect results
######################################################################

print(aggr)

hist(
  aggr$dprime,
  main = "Distribution of d-prime",
  xlab = "d-prime"
)


######################################################################
# One-sample t-test against zero
######################################################################

dprime_test <- t.test(
  aggr$dprime,
  mu = 0,
  alternative = "two.sided"
)

print(dprime_test)


######################################################################
# Descriptive statistics
######################################################################

describe(aggr$dprime)


######################################################################
# Cohen's d for a one-sample design
######################################################################

cohens_d <- mean(aggr$dprime, na.rm = TRUE) /
  sd(aggr$dprime, na.rm = TRUE)

cohens_d


######################################################################
# Results by speaker
######################################################################

aggr_speaker <- PilotStudy %>%
  group_by(ID, Speaker) %>%
  summarise(
    n_masked = sum(Mask == "Masked" & !is.na(Response)),
    n_unmasked = sum(Mask == "Unmasked" & !is.na(Response)),
    
    hits = sum(
      Mask == "Masked" & Response == "1",
      na.rm = TRUE
    ),
    
    false_alarms = sum(
      Mask == "Unmasked" & Response == "1",
      na.rm = TRUE
    ),
    
    .groups = "drop"
  ) %>%
  mutate(
    hit_rate = (hits + 0.5) / (n_masked + 1),
    fa_rate = (false_alarms + 0.5) / (n_unmasked + 1),
    
    dprime = qnorm(hit_rate) - qnorm(fa_rate)
  )

print(aggr_speaker)