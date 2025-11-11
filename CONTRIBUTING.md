# Contributing to HPR-LP

Thank you for your interest in contributing to HPR-LP!

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages. This leads to more readable messages that are easy to follow when looking through the project history.

### Commit Message Format

Each commit message consists of a **header**, a **body**, and a **footer**.

```
<type>(<scope>): <subject>

<body>

<footer>
```

The **header** is mandatory and the **scope** of the header is optional.

### Type

Must be one of the following:

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to our CI configuration files and scripts
- **chore**: Other changes that don't modify src or test files
- **revert**: Reverts a previous commit

### Scope

The scope should be the name of the module/component affected (as perceived by the person reading the changelog).

Examples:
- `algorithm`
- `kernels`
- `utils`
- `docs`

### Subject

The subject contains a succinct description of the change:

- Use the imperative, present tense: "change" not "changed" nor "changes"
- Don't capitalize the first letter
- No dot (.) at the end

### Body

The body should include the motivation for the change and contrast this with previous behavior.

### Footer

The footer should contain any information about **Breaking Changes** and is also the place to reference GitHub issues that this commit closes.

### Examples

```
feat(algorithm): add new HPR solver method

Implement a new high-performance solver for large-scale LP problems
that reduces memory footprint by 30%.

Closes #42
```

```
fix(kernels): correct matrix indexing bug

Fix off-by-one error in kernel computation that caused incorrect
results for edge cases.

Fixes #156
```

```
docs: update installation instructions

Add Julia 1.9+ requirement and clarify dependency installation steps.
```

### Using the Commit Template

To configure git to use the provided commit message template:

```bash
git config commit.template .gitmessage
```

This will automatically populate your commit message with the template when you run `git commit`.

## Pull Request Process

1. Fork the repository and create your branch from `main`
2. Make your changes following the coding standards
3. Add tests if applicable
4. Update documentation as needed
5. Ensure all tests pass
6. Write clear, conventional commit messages
7. Submit a pull request

## Questions?

Feel free to open an issue if you have any questions about contributing!
