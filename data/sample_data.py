

import pandas as pd
import numpy as np

# ─── FAKE NEWS SAMPLES ────────────────────────────────────────────────────────
FAKE_NEWS = [
    "SHOCKING: Scientists Discover That Vaccines Cause Autism, Government Covering Up Truth",
    "EXCLUSIVE: Bill Gates Admits Microchips Inserted in COVID Vaccines to Track Population",
    "Breaking: 5G Towers Secretly Installed Mind-Control Devices, Whistleblower Reveals",
    "ALERT: Global Elites Planning to Replace World Population by 2030, Leaked Documents Show",
    "Confirmed: Moon Landing Was Faked in Hollywood Studio by NASA, Lost Footage Proves It",
    "Bombshell: Climate Change Is Biggest Hoax in History, Top Scientists Silenced by Big Oil",
    "Secret Report: Tap Water Contains Chemicals to Make Population Docile and Compliant",
    "REVEALED: Chemtrails Are Real — Government Admits to Spraying Toxins on Citizens",
    "Breaking News: Flat Earth Theory Confirmed by Antarctic Explorer, NASA Panics",
    "World Government Plans Fake Alien Invasion to Seize Total Control of Nations",
    "Miracle Cure for Cancer Suppressed by Pharma Companies to Maintain Profits, Doctor Claims",
    "URGENT: New Study Finds Smartphones Cause Brain Cancer, Tech Giants Paying to Hide It",
    "Leaked: Obama Born in Kenya, Supreme Court Finally Set to Overturn 2008 Election",
    "Scientists Say Drinking Bleach Kills Coronavirus, Media Refusing to Report It",
    "EXCLUSIVE: Reptilian Elite Control Hollywood, Banks, and Government — Insider Escapes",
    "Soros Funds Antifa Army to Overthrow Government, Senate Hearings Suppressed",
    "Breaking: FBI Whistleblower Reveals Deep State Plot to Assassinate Conservative Leaders",
    "Confirmed: Election Machines Programmed to Flip Votes, IT Expert Exposes Democrat Fraud",
    "World Health Organization Caught Fabricating Pandemic to Push Global Vaccine Agenda",
    "ALERT: Famous Celebrity Died of Vaccine Side Effects, Death Certificate Altered by Hospital",
    "Scientists baffled as giant hole opens in the Sun, threatening Earth's magnetic field collapse",
    "New study proves eating chocolate for breakfast boosts IQ by 20 points, school boards ignore it",
    "Leaked Pentagon files reveal US military has been fighting alien forces in underground bases",
    "Top doctors warn: common household spice cures all cancers but FDA bans discussion of it",
    "BREAKING: Supreme Court secretly voted to abolish all elections, MSM blackout in effect",
]

# ─── REAL NEWS SAMPLES ────────────────────────────────────────────────────────
REAL_NEWS = [
    "Federal Reserve Raises Interest Rates by 0.25 Percent Amid Ongoing Inflation Concerns",
    "UN Climate Summit Reaches Agreement on Carbon Emission Reduction Targets for 2030",
    "Scientists Publish New Research on mRNA Vaccine Efficacy Against Omicron Variant",
    "Congressional Budget Office Projects $1.4 Trillion Deficit for Upcoming Fiscal Year",
    "NASA's James Webb Telescope Captures Deepest Infrared Image of Universe Ever Taken",
    "Tech Giants Face Antitrust Scrutiny as EU Prepares New Digital Markets Act Enforcement",
    "World Health Organization Declares End to COVID-19 Public Health Emergency",
    "Stock Markets Rally After Better-than-Expected Jobs Report Released by Labor Department",
    "Scientists Confirm Discovery of New Species of Deep-Sea Fish Off Pacific Coast",
    "Supreme Court Hears Arguments on Voting Rights Act in Closely Watched Case",
    "Ukraine Receives Additional Military Aid Package from Western Alliance Partners",
    "New Study Links Air Pollution to Increased Risk of Cognitive Decline in Elderly",
    "G7 Nations Agree on Framework to Regulate Artificial Intelligence Development",
    "Census Bureau Reports Slight Decline in US Poverty Rate Following Pandemic Recovery",
    "Researchers Develop New Battery Technology That Could Double Electric Vehicle Range",
    "International Court of Justice Issues Ruling in Maritime Territorial Dispute",
    "European Central Bank Holds Rates Steady as Inflation Shows Signs of Easing",
    "Congress Passes Bipartisan Infrastructure Bill with Funding for Roads and Bridges",
    "Pfizer Reports Strong Quarterly Earnings Driven by Vaccine and Antiviral Sales",
    "NASA Confirms Mars Rover Successfully Collected Rock Samples for Future Return Mission",
    "IMF Upgrades Global Growth Forecast to 3.2 Percent for Current Fiscal Year",
    "Federal Court Rules Against Tech Platform's Content Moderation Policy in First Amendment Case",
    "New research from MIT shows quantum computing achieves breakthrough in error correction",
    "WHO releases updated guidelines on antibiotic resistance prevention strategies",
    "Department of Energy announces $3 billion investment in advanced nuclear reactor technology",
]

# ─── PROPAGANDA SAMPLES ───────────────────────────────────────────────────────
PROPAGANDA = [
    "Every true patriot must stand against the corrupt elites who are destroying our nation",
    "The enemy within our borders is far more dangerous than any foreign threat we face",
    "They want to take away your freedom, your guns, your religion — we must resist now",
    "Our glorious leader is under attack by jealous criminals who fear his unstoppable power",
    "Join the movement of real citizens who refuse to bow down to the globalist agenda",
    "The mainstream media is your enemy — they lie constantly to keep you enslaved",
    "We are at war with those who seek to destroy our culture and replace our people",
    "Only by purging the traitors in our government can we restore our nation's greatness",
    "The immigrants are stealing your jobs, your homes, your future — we must act now",
    "Stand with us or stand with the terrorists — there is no middle ground in this fight",
    "Our children are being indoctrinated in schools by radical ideologues who hate America",
    "The election was stolen from you — join millions who demand justice be served today",
    "Western civilization faces its greatest threat from within — cowardly liberals must be stopped",
    "Every dollar you pay in taxes goes to fund programs that destroy the fabric of society",
    "Real Americans do not need permission to exercise rights that God himself granted to us",
    "If you are not outraged, you are not paying attention to what they are doing to our country",
    "The bankers, the media, the politicians — they are all connected in a vast criminal conspiracy",
    "History will remember those who fought bravely and those who cowered in silence today",
    "Your vote does not matter unless you vote to stop the radical agenda destroying our future",
    "We built this country with blood and sacrifice — we will not let them tear it apart",
    "Patriots must rise up against the socialist agenda being forced upon innocent Americans",
    "The deep state operatives have infiltrated every level of government to destroy liberty",
    "Our ancestors would weep to see what cowards have done to the nation they built for us",
    "Time is running out — if we don't act now, freedom will be lost for generations to come",
    "Wake up before it is too late — they are coming for your guns, your children, your soul",
]

# ─── PROPAGADA TECHNIQUE LABELS ───────────────────────────────────────────────
PROPAGANDA_TECHNIQUES = {
    0: "Appeal to Fear",
    1: "Loaded Language",
    2: "Name Calling / Labeling",
    3: "Bandwagon",
    4: "Black-and-White Fallacy",
    5: "Glittering Generalities",
    6: "Repetition",
    7: "Exaggeration/Minimisation",
}

PROPAGANDA_TECHNIQUE_MAP = {
    "Every true patriot must stand against the corrupt elites who are destroying our nation": 5,
    "The enemy within our borders is far more dangerous than any foreign threat we face": 0,
    "They want to take away your freedom, your guns, your religion — we must resist now": 0,
    "Our glorious leader is under attack by jealous criminals who fear his unstoppable power": 2,
    "Join the movement of real citizens who refuse to bow down to the globalist agenda": 3,
    "The mainstream media is your enemy — they lie constantly to keep you enslaved": 2,
    "We are at war with those who seek to destroy our culture and replace our people": 1,
    "Only by purging the traitors in our government can we restore our nation's greatness": 2,
    "The immigrants are stealing your jobs, your homes, your future — we must act now": 0,
    "Stand with us or stand with the terrorists — there is no middle ground in this fight": 4,
}


def get_dataset():
    """Return a combined DataFrame of all samples with labels."""
    records = []

    for text in FAKE_NEWS:
        records.append({"text": text, "label": "fake", "label_id": 0})

    for text in REAL_NEWS:
        records.append({"text": text, "label": "real", "label_id": 1})

    for text in PROPAGANDA:
        records.append({"text": text, "label": "propaganda", "label_id": 2})

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def get_binary_dataset():
    """Return fake vs real only (binary classification)."""
    records = []
    for text in FAKE_NEWS:
        records.append({"text": text, "label": "FAKE", "label_id": 0})
    for text in REAL_NEWS:
        records.append({"text": text, "label": "REAL", "label_id": 1})
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def get_propaganda_dataset():
    """Return propaganda vs non-propaganda (binary classification)."""
    records = []
    for text in PROPAGANDA:
        records.append({"text": text, "label": "PROPAGANDA", "label_id": 1})
    for text in REAL_NEWS[:25]:
        records.append({"text": text, "label": "NOT_PROPAGANDA", "label_id": 0})
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = get_dataset()
    print(df["label"].value_counts())
    print(df.head())
