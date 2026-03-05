#ifndef YLLANG_DISTRIBUTED_INFO_H_
#define YLLANG_DISTRIBUTED_INFO_H_

namespace yllang {
class DistributedInfo {
 public:
  DistributedInfo(const int rank, int size) noexcept : m_rank_(rank), m_size_(size) {}

  auto GetRank() const -> int { return m_rank_; }
  auto GetSize() const -> int { return m_size_; }

  void SetRank(int rank) { m_rank_ = rank; }
  void SetSize(int size) { m_size_ = size; }

 private:
  int m_rank_;
  int m_size_;
};

inline static thread_local DistributedInfo distributed_info(0, 1);

inline auto SetDistributedInfo(int rank, int size) -> void {
  distributed_info.SetRank(rank);
  distributed_info.SetSize(size);
}

inline auto GetDistributedInfo() -> const DistributedInfo & { return distributed_info; }

}  // namespace yllang

#endif  // YLLANG_DISTRIBUTED_INFO_H_