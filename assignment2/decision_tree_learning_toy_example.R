# Decision Tree Learning: a toy example
# -------------------------------------

# CART
# ----

# Use rpart library.
library(rpart)

# Load data into session.
train <- read.csv("assignment2/playTennis.csv")

# Formula used to compute the relation.
formula  = Class ~ outlook + temperature + humidity + wind

# Set rpart options; in this case, split if more than minsplit observations are left in a node
# and if the lack of fit is reduced by at least cp by the split.
option <- rpart.control(minsplit = 1, cp = 0)

# Construct the decision tree.
tree <- rpart(formula, data=train, control=option)

# Output the tree.
print(tree)
plot(tree)
text(tree)

# Questions
# ---------

# We cannot produce the ID3 tree using CART, since the latter can only construct binary trees.
# Using minsplit=10 and cp=0 guarantees the split will be performed only
#  if the node has at least 10 observations (and, trivially, at least decreases lack of fit by 0).
# ID3 splits each node separately, but the same attribute can be used multiple times.
# Using cp=1 means that splits only occur if they decrease the lack of fit by 1.
# CART only computes binary trees; ID3 does not.
# The ID3 tree can always be reproduced functionally (by binarizing each attribute sufficiently).
# CART trees are deeper than ID3 trees because they are binary whereas ID3 trees are not.
# It is not always trivial to "merge" nodes like that,
#  since binarization does not always happen on the same attribute in succession.