/**
 * @file info.h
 * @brief Defines distributed runtime information and global accessors.
 */

#ifndef YLLANG_DISTRIBUTED_INFO_H_
#define YLLANG_DISTRIBUTED_INFO_H_

namespace yllang {

/**
 * @brief Holds rank and world size for distributed execution.
 *
 * This class stores the rank (process ID) and total number of processes
 * in a distributed environment. It is used to support tensor parallelism
 * and other collective operations.
 */
class DistributedInfo {
 public:
  /**
   * @brief Constructs a DistributedInfo object.
   *
   * @param rank  Rank of the current process (0-based).
   * @param size  Total number of processes.
   */
  DistributedInfo(const int rank, int size) noexcept : m_rank_(rank), m_size_(size) {}

  /// Returns the rank of the current process.
  auto GetRank() const -> int { return m_rank_; }

  /// Returns the total number of processes.
  auto GetSize() const -> int { return m_size_; }

  /// Sets the rank.
  void SetRank(int rank) { m_rank_ = rank; }

  /// Sets the world size.
  void SetSize(int size) { m_size_ = size; }

 private:
  int m_rank_;  ///< Process rank (0 .. size-1).
  int m_size_;  ///< Total number of processes.
};

/**
 * @brief Returns a reference to the global DistributedInfo singleton.
 *
 * The instance is created on first use and is thread‑safe (C++11 magic statics).
 *
 * @return DistributedInfo& The global distributed info object.
 */
inline static auto GetDistributedInfo() -> DistributedInfo & {
  static DistributedInfo instance(0, 1);  // Default: single process.
  return instance;
}

/**
 * @brief Convenience function to set the global distributed info.
 *
 * @param rank Rank of the current process.
 * @param size Total number of processes.
 */
inline auto SetDistributedInfo(int rank, int size) -> void {
  GetDistributedInfo().SetRank(rank);
  GetDistributedInfo().SetSize(size);
}

}  // namespace yllang

#endif  // YLLANG_DISTRIBUTED_INFO_H_