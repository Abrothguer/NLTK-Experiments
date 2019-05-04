"""
    Samples
"""
# pylint: disable=C0103

# Sample text - Excerpt from Do not go gentle into that good nigth by Dylan Thomas
sample = """Do not go gentle into that good night.
            Old age should burn and rave at close of day.
            Rage, rage against the dying of the light.

            Though wise men at their end know dark is right.
            Because their words had forked no lightning they.
            Do not go gentle into that good night."""
sample = " ".join(sample.replace('\n', '').split())

# Sample tweet
sample_tw = """.@SpaceX is now targeting May 1 at 3:59am ET for the next
                cargo launch to the @Space_Station. Onboard will be more
                than 5,500 pounds of @ISS_Research, supplies and hardware
                for crew members living and working on our orbiting outpost.
                Details: https://go.nasa.gov/2GExQpL """
sample_tw = " ".join(sample_tw.replace('\n', '').split())

# Sample contractions
sample_ct = """ Money, get back.
                I'm all right Jack keep your hands off of my stack.
                Money, it's a hit.
                Don't give me that do goody good bullshit.
                I'm in the high-fidelity first class traveling set.
                And I think I need a Lear jet."""
sample_ct = " ".join(sample_ct.replace('\n', '').split())

# Sample quotes

quote_1 = "The cake is a lie, but the cherry is not"
quote_2 = "A wizard is never late"
quote_3 = "The meeting was boring"
quote_4 = "Most of the Top 10 companies in the world are in the tech business"
quote_5 = "The sentence of death"
quote_6 = "The shores picture"
quote_7 = "The picture of costlines is beautiful"

# Wrong sentences
wrong_1 = "Is the children singing"
wrong_2 = "The doctor were right"
