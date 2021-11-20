TEST_FILES=$(find ./tests ./firecannon -name "*.test.py")

for x in $TEST_FILES; do
  python "$x";
  done
