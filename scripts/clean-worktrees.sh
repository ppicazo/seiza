#!/usr/bin/env bash
# Report the disk each Seiza checkout uses, and reclaim it safely.
#
#   scripts/clean-worktrees.sh              report only (the default)
#   scripts/clean-worktrees.sh --targets    offer to delete the build output
#                                           (target/, seiza-py/target/) of
#                                           checkouts nothing is using
#   scripts/clean-worktrees.sh --remove     offer to remove linked worktrees
#                                           that are idle, clean and merged
#
# Every deletion asks first. A checkout is "in use" when a process has its
# working directory inside it or holds one of its Cargo build locks; nothing
# in use is ever touched. Worktrees are removed with `git worktree remove`
# (never --force), so Git itself refuses one with uncommitted changes.
# Worktrees the Claude app made under .claude/worktrees are better left to
# the app (Settings > This computer > Storage), which also keeps the session
# that owns them in step; this script only offers them when the session is
# long idle.
set -euo pipefail

mode=report
idle_days=7
base=main
while [ $# -gt 0 ]; do
  case "$1" in
    --targets) mode=targets ;;
    --remove) mode=remove ;;
    --idle-days) idle_days="$2"; shift ;;
    --base) base="$2"; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

repo=$(git rev-parse --path-format=absolute --git-common-dir)
repo=$(dirname "$repo")
# Run from outside every checkout, so this script never counts as using one.
cd /

# Worktrees nested inside the main checkout are counted on their own rows.
size() { du -sh --exclude="$repo/.claude/worktrees" "$1" 2>/dev/null | cut -f1; }

# True when a process works inside $1 or holds a Cargo lock under it. In the
# main checkout only builds count (cargo, rustc, maturin, the linker), since a
# shell or editor sits there most of the time; in a linked worktree any
# process does, as a Claude session runs in it.
in_use() {
  local dir=$1 link pid
  for link in /proc/[0-9]*/cwd; do
    case "$(readlink "$link" 2>/dev/null)/" in
      "$dir"/*)
        [ "$dir" != "$repo" ] && return 0
        pid=${link#/proc/}; pid=${pid%/cwd}
        case "$(cat "/proc/$pid/comm" 2>/dev/null)" in
          cargo|rustc|maturin|cc|ld|ld.lld|lld|clang|gcc|build-script-*) return 0 ;;
        esac
        ;;
    esac
  done
  local lock
  for lock in "$dir"/target/*/.cargo-lock "$dir"/seiza-py/target/*/.cargo-lock; do
    [ -e "$lock" ] && command -v fuser >/dev/null && fuser -s "$lock" 2>/dev/null && return 0
  done
  return 1
}

# Days since anything in the checkout's sources (not its build output) changed.
idle_for() {
  local newest
  newest=$(find "$1" -path "$1/target" -prune -o -path "$1/seiza-py/target" -prune \
    -o -path "$1/.git" -prune -o -path "$1/.claude/worktrees" -prune -o -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
  echo $(( ( $(date +%s) - ${newest%.*} ) / 86400 ))
}

confirm() { local reply; read -r -p "$1 [y/N] " reply </dev/tty; [ "$reply" = y ] || [ "$reply" = Y ]; }

git -C "$repo" worktree prune --dry-run -v
printf '%-58s %6s %6s %5s %6s %s\n' CHECKOUT TOTAL TARGET IDLE STATE NOTES

git -C "$repo" worktree list --porcelain | awk '/^worktree /{print substr($0, 10)}' | while read -r dir; do
  [ -d "$dir" ] || continue
  branch=$(git -C "$dir" symbolic-ref --quiet --short HEAD || echo "(detached)")
  dirty=$(git -C "$dir" status --porcelain --untracked-files=normal | wc -l)
  unmerged=$(git -C "$dir" rev-list --count "$base..HEAD" 2>/dev/null || echo "?")
  unpushed=$(git -C "$dir" rev-list --count HEAD --not --remotes 2>/dev/null || echo "?")
  idle=$(idle_for "$dir")
  state=idle; in_use "$dir" && state=ACTIVE
  build=0
  for t in "$dir/target" "$dir/seiza-py/target"; do [ -d "$t" ] && build=1; done
  notes="$branch; $dirty changed files; $unmerged commits not in $base; $unpushed unpushed"
  printf '%-58s %6s %6s %4sd %6s %s\n' "${dir#"$HOME"/}" "$(size "$dir")" \
    "$( [ $build = 1 ] && size "$dir/target" || echo -)" "$idle" "$state" "$notes"

  [ "$state" = ACTIVE ] && continue
  case "$mode" in
    targets)
      [ $build = 1 ] || continue
      if confirm "  delete build output in $dir (target/, seiza-py/target/)?"; then
        rm -rf -- "$dir/target" "$dir/seiza-py/target"
        echo "  deleted"
      fi
      ;;
    remove)
      [ "$dir" = "$repo" ] && continue
      if [ "$dirty" != 0 ] || [ "$unmerged" != 0 ] || [ "$unpushed" != 0 ] || [ "$idle" -lt "$idle_days" ]; then
        echo "  kept: holds unique work or was used in the last $idle_days days"
        continue
      fi
      if confirm "  remove worktree $dir (branch $branch is kept)?"; then
        git -C "$repo" worktree remove -- "$dir" && echo "  removed"
      fi
      ;;
  esac
done
df -h "$repo" | tail -1 | awk '{print "free on " $6 ": " $4}'
