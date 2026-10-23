# Error Analysis

Notes on where the three models go wrong on the GoEmotions test set (4,590 single-label examples, 28 classes). Overall results are in the main [README](README.md); this is just the "why" part. Everything here comes from the notebook outputs.

## Short version

- All three models do well on classes with obvious trigger words (gratitude, love, amusement) and badly on vague ones (realization, annoyance, relief, disappointment).
- DistilBERT is better on most classes, but not on `neutral`, where TF-IDF is far ahead (0.566 vs 0.371 F1). That one class is a big reason TF-IDF still wins on accuracy and weighted-F1.
- Most mistakes are between emotions that sit close together (anger/annoyance, confusion/curiosity) or into broad labels like `approval`.
- Some of the "confident errors" look more like questionable labels than model failures.

## Per-class F1

Sorted by DistilBERT F1. Support is the number of test examples.

| Class | TF-IDF | DistilBERT | BiLSTM | Support |
|-------|:------:|:----------:|:------:|--------:|
| gratitude | 0.920 | 0.909 | 0.904 | 260 |
| love | 0.772 | 0.808 | 0.797 | 160 |
| amusement | 0.759 | 0.800 | 0.783 | 186 |
| fear | 0.636 | 0.681 | 0.657 | 65 |
| admiration | 0.612 | 0.675 | 0.586 | 348 |
| remorse | 0.680 | 0.673 | 0.672 | 44 |
| sadness | 0.478 | 0.597 | 0.450 | 102 |
| joy | 0.482 | 0.576 | 0.576 | 93 |
| optimism | 0.574 | 0.553 | 0.605 | 107 |
| curiosity | 0.392 | 0.533 | 0.521 | 176 |
| embarrassment | 0.318 | 0.526 | 0.407 | 23 |
| surprise | 0.384 | 0.495 | 0.443 | 87 |
| anger | 0.392 | 0.485 | 0.433 | 131 |
| desire | 0.429 | 0.477 | 0.421 | 56 |
| disgust | 0.497 | 0.462 | 0.451 | 76 |
| nervousness | 0.250 | 0.409 | 0.308 | 12 |
| confusion | 0.338 | 0.403 | 0.304 | 97 |
| caring | 0.234 | 0.388 | 0.253 | 86 |
| excitement | 0.333 | 0.383 | 0.257 | 57 |
| approval | 0.293 | 0.380 | 0.325 | 236 |
| neutral | 0.566 | 0.371 | 0.392 | 1606 |
| disapproval | 0.319 | 0.371 | 0.282 | 195 |
| pride | 0.364 | 0.333 | 0.235 | 7 |
| disappointment | 0.224 | 0.298 | 0.225 | 88 |
| annoyance | 0.211 | 0.267 | 0.154 | 194 |
| relief | 0.133 | 0.207 | 0.071 | 7 |
| realization | 0.207 | 0.188 | 0.194 | 89 |
| grief | 0.000 | 0.143 | 0.286 | 2 |

![Per-class F1](figures/per_class_f1.png)

DistilBERT is the best of the three on 19 of the 28 classes. The biggest gains over TF-IDF are on embarrassment (+0.21), nervousness (+0.16), caring (+0.15), curiosity (+0.14) and sadness (+0.12). TF-IDF is still ahead on gratitude, remorse, disgust, pride and neutral. The BiLSTM only wins on optimism and grief, and grief has 2 test examples, so I wouldn't read anything into that.

The ones that stand out to me are realization and annoyance. They're low for every model even though they have a decent number of test examples (89 and 194), so it isn't just a small-sample problem.

## Does class size explain it?

Partly. The plot below shows test F1 against training examples per class (log scale).

![F1 vs class frequency](figures/f1_vs_support.png)

Bigger classes generally score higher, but there's a lot of scatter. Some classes with a few hundred training examples reach 0.6-0.7 while some much bigger ones stay under 0.3. And `neutral`, the biggest class by far (~12,800 training examples), is nowhere near the top for any model. It's a catch-all with no consistent vocabulary, so having lots of data doesn't help much. All models were trained with class weights, so the rare-class results already include that fix.

## Confusion matrix (DistilBERT)

![DistilBERT confusion matrix](figures/confusion_distilbert.png)

Most confused pairs (fraction of the true class predicted as the other class):

| True | Predicted as | Fraction |
|------|--------------|:--------:|
| grief | anger | 0.500 |
| pride | approval | 0.286 |
| confusion | curiosity | 0.216 |
| realization | approval | 0.146 |
| anger | annoyance | 0.145 |
| annoyance | anger | 0.144 |
| pride | admiration | 0.143 |
| pride | disapproval | 0.143 |
| relief | admiration / approval / caring / joy | 0.143 each |

A few things:

- anger and annoyance get mixed up in both directions at about the same rate, which isn't surprising since they're close in meaning. Confusion going to curiosity makes sense too, since a lot of questions are both.
- `approval` seems to act as a dump class. Pride and realization both leak into it.
- The grief, pride and relief rows are basically noise. Grief has 2 test examples, pride and relief have 7, so one prediction moves the fraction by 14-50 points. The `grief → anger` 0.500 is literally one example out of two.
- The `neutral` row is spread over lots of different classes and the diagonal is faint. My guess is that class weighting (which down-weights neutral on purpose) pushes DistilBERT to over-assign emotions to neutral text. I didn't test this, so it's only a guess. A run without class weights would settle it.

## Confident mistakes

Top 15 wrong DistilBERT predictions by confidence (all 0.98-0.99):

| Text | True | Predicted |
|------|------|-----------|
| Thanks for further justifying my support for Tulsi, Daily Gross. | neutral | gratitude |
| Yeah thank you you ungrateful bast..... | anger | gratitude |
| Thank you! [NAME] seems to realize how toxic this fanbase can be also.. | neutral | gratitude |
| Your comments have been removed. This is a community for support. ... Thank you f... | neutral | gratitude |
| [NAME] looks veeeery nervous | neutral | nervousness |
| Former [NAME]? I've been scared already by [NAME] who were former d-line | curiosity | fear |
| I wish that too. | optimism | desire |
| Have you ever consider knocking three times ... stalker vibe is terrifying. | annoyance | fear |
| Sadly enough I was thinking the same. | remorse | sadness |
| Yep California, I would've been surprised if they cleared him. | approval | surprise |
| It's real enough sadly. Kin have been around for awhile in their own corners of the net. | neutral | sadness |
| LOVE THIS!! | admiration | love |
| so glad we picked this guy up. was absolutely terrified when we played against him. | joy | fear |
| [NAME] loves rui too afterall | neutral | love |
| Hope all is well now xo | caring | optimism |

Most of these are the model latching onto a single word: "thank" gives gratitude, "nervous" gives nervousness, "scared/terrifying/terrified" gives fear, "sadly" gives sadness, "love" gives love. The sarcastic "thank you you ungrateful bast....." is read as gratitude, and "so glad we picked this guy up. was absolutely terrified..." is clearly positive but the fear word wins.

Also, quite a few of the gold labels are arguable. "[NAME] looks veeeery nervous" is labelled neutral, and "LOVE THIS!!" is admiration. I wouldn't call those clear model errors, so part of the ceiling here is label noise. Another thing I noticed is that all 15 sit at 0.98-0.99 confidence, so softmax confidence wouldn't help flag them.

## What TF-IDF picks up on

Top features by logistic regression coefficient for a few classes:

- gratitude: thanks, thank, thank you, welcome, glad, thanks for, congrats, appreciate
- amusement: lol, haha, funny, lmao, hilarious, fun, hahaha, laughed
- fear: afraid, scared, scary, terrible, terrifying, horrible, dangerous, fear
- grief: died, rip, dead, death, your loss, bro, deaths, very sorry
- disgust: disgusting, awful, worst, weird, ugly, nasty, worse, gross
- confusion: confused, not sure, why, or, confusing, don know, idk, confusion
- annoyance: annoying, stupid, annoyed, shit, idiots, fucking, pissed, fuck
- anger: fuck, hate, fucking, angry, stupid, asshole, bitch, stop

The easy classes are basically explicit emotion words, which is why TF-IDF does so well on gratitude and amusement and why DistilBERT doesn't add much there. Annoyance and anger share words (stupid, fucking, fuck), which fits the confusion between them. A few features look like noise: "bro" under grief and "or"/"why" under confusion, probably from very few training examples.

## Limitations

- Single run, single seed, no confidence intervals, so small F1 differences (like BiLSTM vs TF-IDF macro-F1, 0.428 vs 0.421) could just be noise.
- The confusion matrix and the confident-error list are only for DistilBERT. I haven't looked at the other two models the same way.
- Only single-label comments are used, which removes some of the ambiguity of the full dataset.
- The explanations above (class weighting and neutral, label noise) are my reading of the outputs, not tested with ablations.
