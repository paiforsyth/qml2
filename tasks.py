from invoke import task

SRC_DIR = "qml"
TEST_DIR = "qml"


@task
def mypy(c):
    c.run(f"mypy {SRC_DIR}")

@task
def test(c):
    c.run(f"pytest --failed-first {SRC_DIR}", pty=True)

@task
def black(c):
    c.run(f"black {SRC_DIR} {TEST_DIR}")


@task
def black_check(c):
    c.run(f"black --check {SRC_DIR} {TEST_DIR}")

@task
def autoflake(c):
    c.run(f"autoflake --recursive --in-place --remove-all-unused-imports --remove-unused-variables {SRC_DIR} {TEST_DIR}")


@task
def autoflake_check(c):
    c.run(f"autoflake --recursive --check --remove-all-unused-imports --remove-unused-variables {SRC_DIR} {TEST_DIR}")


@task
def isort(c):
    c.run(f"isort {SRC_DIR} {TEST_DIR}")


@task
def isort_check(c):
    c.run(f"isort --check {SRC_DIR} {TEST_DIR}")

# @task
# def lint_imports(c):
#     c.run("lint-imports")


@task
def fix(c):
    autoflake(c)
    isort(c)
    black(c)

@task
def check(c):
    autoflake_check(c)
    isort_check(c)
    black_check(c)
    # lint_imports(c)