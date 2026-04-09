Student mental health is now a major issue in
today’s schools. The usual ways of checking mental health,
like regular surveys or students reporting their feelings, are not
very reliable and can’t be used effectively in real-time across a
whole classroom. This paper introduces an AI-based system for
monitoring student mental well-being. It’s a desktop program
that runs locally on school computers and uses computer vision
and deep learning to check how students are feeling during
classes. The system uses three main tools: the Viola-Jones Haar
Cascade for quickly detecting faces in real time, the Local Binary
Patterns Histogram (LBPH) method for recognizing individual
students, and a mini-XCEPTION neural network trained on the
FER2013 dataset to identify seven different emotions. It also
has a system that turns these detected emotions into a score
between 0 and 100, and then categorizes students into four
groups: THRIVING, GOOD, OKAY, and NEEDS ATTENTION.
All data is stored locally to keep it private. The system has been
tested and shows an 89% accuracy rate in identifying students
based on eight photos each. Teachers get a colorful dashboard
showing the class’s emotional state, and counselors can access
detailed reports on each student’s mental health over time.
