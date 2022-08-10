# Introducing slangID2.0

In a nutshell: The slangID project tries to detect slang phrases. Something literally no one asked for...

You can train a selection of Machine Learning Models, and print out a test set of phrases with the **DEMO** button.
Or you can type a phrase and see what type it is identified as. All the models are pre-trained, but you can re-train if needed.

# Challenges

Due to a lack of data, the results, regardless of the classifier used, are not good enough right now.
 Unknown words are also an issue since the dataset is tiny.
 
# How to run slangID2.0

1. Install Python **3.9** or later (3.8 and 3.10 is probably fine too, I used 3.9.12).
2. Install the required packages by running `pip install -r requirements.txt` in your shell of choice. Make sure you are in the project directory.
3. And then run `python slangID2.0_Windows.py` or `python3 slangID2.0_Linux.py` (the difference between both versions is just the font size on some labels and buttons).

**Note:** It might take a while to load. Be patient.

# Screenshot

![slangID2.0](misc/slangID2.0_screenshot.png)

# Source of the data

Most of the phrases are hand-picked and come from archive.org's [Twitter Stream of June 6th](https://archive.org/details/archiveteam-twitter-stream-2021-06).

Some come from me personally, which you might recognize due to their sad and depressing nature.

# Recognition of Open Source use

* scikit-learn
* pandas
