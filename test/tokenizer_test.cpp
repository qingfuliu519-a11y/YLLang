/**
 * @file tokenizer_test.cpp
 * @brief Unit tests for the Tokenizer class.
 */
#include "tokenizer/tokenizer.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "config/config.h"
#include "models/model.h"

namespace yllang::test {

using json = nlohmann::json;

/**
 * @brief Loads test data from a JSON file.
 *
 * The JSON file must contain an array of test cases. Each test case may
 * contain fields such as "prompt", "tokens", "original_sentence", etc.
 *
 * @param filename Path to the JSON file.
 * @return json The parsed JSON array.
 * @throws std::runtime_error if file cannot be opened or data is not an array.
 */
auto LoadTestData(const std::string &filename) -> json {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open test data file: " + filename);
  }
  json data;
  file >> data;
  if (!data.is_array()) {
    throw std::runtime_error("Test data must be a JSON array");
  }
  return data;
}

/**
 * @brief Tests that the tokenizer can be loaded successfully from a JSON file.
 *
 * @param tokenizer_json_path Path to the tokenizer JSON file.
 */
auto TestTokenizerLoad(const std::string &tokenizer_json_path) -> void {
  std::cout << "\n=== TestTokenizerLoad ===\n";
  auto tokenizer = yllang::FromJSON(tokenizer_json_path);
  std::cout << "Checking tokenizer != nullptr\n";
  assert(tokenizer != nullptr);
  std::cout << "Checking !tokenizer->Empty()\n";
  assert(!tokenizer->Empty());
  std::cout << "Tokenizer loaded successfully.\n";
  std::cout << "--- TestTokenizerLoad passed ---\n";
}

/**
 * @brief Tests that encoding a prompt yields the expected token sequence.
 *
 * @param tokenizer_json_path Path to the tokenizer JSON file.
 * @param test_data_path      Path to the JSON test data file.
 */
auto TestTokenizerEncode(const std::string &tokenizer_json_path, const std::string &test_data_path) -> void {
  std::cout << "\n=== TestTokenizerEncode ===\n";
  auto tokenizer = yllang::FromJSON(tokenizer_json_path);
  assert(tokenizer && !tokenizer->Empty());

  auto test_data = LoadTestData(test_data_path);
  for (const auto &item : test_data) {
    std::string prompt = item.value("prompt", "");
    std::vector<int32_t> expected_tokens = item.value("tokens", std::vector<int32_t>());
    if (prompt.empty() || expected_tokens.empty()) {
      std::cout << "Skipping invalid test case (empty prompt or tokens).\n";
      continue;
    }
    auto encoded = tokenizer->Encode(prompt);
    std::cout << "Checking encode for prompt: " << prompt << "\n";
    assert(encoded == expected_tokens);
    std::cout << "  OK.\n";
  }
  std::cout << "--- TestTokenizerEncode passed ---\n";
}

/**
 * @brief Tests that decoding a token sequence yields the original prompt.
 *
 * @param tokenizer_json_path Path to the tokenizer JSON file.
 * @param test_data_path      Path to the JSON test data file.
 */
auto TestTokenizerDecode(const std::string &tokenizer_json_path, const std::string &test_data_path) -> void {
  std::cout << "\n=== TestTokenizerDecode ===\n";
  auto tokenizer = yllang::FromJSON(tokenizer_json_path);
  assert(tokenizer && !tokenizer->Empty());

  auto test_data = LoadTestData(test_data_path);
  for (const auto &item : test_data) {
    std::string prompt = item.value("prompt", "");
    std::vector<int32_t> tokens = item.value("tokens", std::vector<int32_t>());
    if (prompt.empty() || tokens.empty()) {
      std::cout << "Skipping invalid test case (empty prompt or tokens).\n";
      continue;
    }
    auto decoded = tokenizer->Decode(tokens);
    std::cout << "Checking decode for tokens: ";
    for (int32_t t : tokens) {
      std::cout << t << " ";
    }
    std::cout << "\n";
    assert(decoded == prompt);
    std::cout << "  OK.\n";
  }
  std::cout << "--- TestTokenizerDecode passed ---\n";
}

/**
 * @brief Tests batch encoding of UserMsg objects against expected token sequences.
 *
 * @param tokenizer_json_path Path to the tokenizer JSON file.
 * @param test_data_path      Path to the JSON test data file.
 */
auto TestTokenizerBatchEncode(const std::string &tokenizer_json_path, const std::string &test_data_path) -> void {
  std::cout << "\n=== TestTokenizerBatchEncode ===\n";
  auto tokenizer = yllang::FromJSON(tokenizer_json_path);
  assert(tokenizer && !tokenizer->Empty());

  auto test_data = LoadTestData(test_data_path);
  std::vector<UserMsg> msgs;
  std::vector<std::vector<int32_t>> expected_batch;
  for (const auto &item : test_data) {
    std::string original = item.value("original_sentence", "");
    std::vector<int32_t> expected_tokens = item.value("tokens", std::vector<int32_t>());
    if (original.empty() || expected_tokens.empty()) {
      std::cout << "Skipping invalid test case (empty original_sentence or tokens).\n";
      continue;
    }
    // Assuming UserMsg constructor takes (role, content)
    std::string user = "user";
    msgs.emplace_back(user, original);
    expected_batch.push_back(expected_tokens);
  }
  auto encoded_batch = tokenizer->EncodeBatch(msgs);
  std::cout << "Checking encoded_batch.size() == " << expected_batch.size() << "\n";
  assert(encoded_batch.size() == expected_batch.size());
  for (size_t i = 0; i < encoded_batch.size(); ++i) {
    std::cout << "Checking batch encode for message " << i << "\n";
    assert(encoded_batch[i] == expected_batch[i]);
    std::cout << "  OK.\n";
  }
  std::cout << "--- TestTokenizerBatchEncode passed ---\n";
}

/**
 * @brief Tests batch decoding of token sequences against expected texts.
 *
 * @param tokenizer_json_path Path to the tokenizer JSON file.
 * @param test_data_path      Path to the JSON test data file.
 */
auto TestTokenizerBatchDecode(const std::string &tokenizer_json_path, const std::string &test_data_path) -> void {
  std::cout << "\n=== TestTokenizerBatchDecode ===\n";
  auto tokenizer = yllang::FromJSON(tokenizer_json_path);
  assert(tokenizer && !tokenizer->Empty());

  auto test_data = LoadTestData(test_data_path);
  std::vector<std::vector<int32_t>> batch_tokens;
  std::vector<std::string> expected_texts;
  for (const auto &item : test_data) {
    std::string prompt = item.value("prompt", "");
    std::vector<int32_t> tokens = item.value("tokens", std::vector<int32_t>());
    if (prompt.empty() || tokens.empty()) {
      std::cout << "Skipping invalid test case (empty prompt or tokens).\n";
      continue;
    }
    batch_tokens.push_back(tokens);
    expected_texts.push_back(prompt);
  }
  auto decoded_batch = tokenizer->DecodeBatch(batch_tokens);
  std::cout << "Checking decoded_batch.size() == " << expected_texts.size() << "\n";
  assert(decoded_batch.size() == expected_texts.size());
  for (size_t i = 0; i < decoded_batch.size(); ++i) {
    std::cout << "Checking batch decode for item " << i << "\n";
    assert(decoded_batch[i] == expected_texts[i]);
    std::cout << "  OK.\n";
  }
  std::cout << "--- TestTokenizerBatchDecode passed ---\n";
}

/**
 * @brief Tests edge cases such as empty strings, empty token sequences, etc.
 *
 * @param tokenizer_json_path Path to the tokenizer JSON file.
 */
auto TestTokenizerEdgeCases(const std::string &tokenizer_json_path) -> void {
  std::cout << "\n=== TestTokenizerEdgeCases ===\n";
  auto tokenizer = yllang::FromJSON(tokenizer_json_path);
  assert(tokenizer && !tokenizer->Empty());

  std::string empty_str;
  auto encoded_empty = tokenizer->Encode(empty_str);
  std::cout << "Encoding empty string, got " << encoded_empty.size() << " tokens.\n";
  assert(encoded_empty.size() >= 0);  // Only check that it doesn't crash.

  // Decode empty token list.
  std::vector<int32_t> empty_ids;
  auto decoded_empty = tokenizer->Decode(empty_ids);
  std::cout << "Decoding empty token list, got: \"" << decoded_empty << "\"\n";
  assert(decoded_empty.empty());

  // Batch encode: empty user message.
  std::vector<UserMsg> empty_msgs;
  empty_msgs.emplace_back("user", "");
  auto empty_batch_enc = tokenizer->EncodeBatch(empty_msgs);
  assert(empty_batch_enc.size() == 1);
  std::cout << "Batch encode empty user msg, token count: " << empty_batch_enc[0].size() << "\n";

  // Batch decode: empty token list.
  std::vector<std::vector<int32_t>> empty_batch_ids = {empty_ids};
  auto empty_batch_dec = tokenizer->DecodeBatch(empty_batch_ids);
  assert(empty_batch_dec.size() == 1);
  assert(empty_batch_dec[0].empty());

  std::cout << "Edge cases handled without crash.\n";
  std::cout << "--- TestTokenizerEdgeCases passed ---\n";
}

}  // namespace yllang::test

/**
 * @brief Main entry point for the tokenizer unit tests.
 *
 * Expects two command line arguments:
 *   1. Path to the tokenizer JSON file.
 *   2. Path to the test data JSON file.
 *
 * @param argc Number of arguments.
 * @param argv Argument list.
 * @return int 0 on success, 1 on usage error.
 */
auto main(int argc, char *argv[]) -> int {
  std::string tokenizer_path = "./tokenizer.json";
  std::string test_data_path =
      yllang::kModelPath + yllang::ModelTypeToString(yllang::ModelType::KQwen306b) + "/tokenized_results.json";

  std::cout << "Running Tokenizer unit tests...\n";

  yllang::test::TestTokenizerLoad(tokenizer_path);
  yllang::test::TestTokenizerEncode(tokenizer_path, test_data_path);
  yllang::test::TestTokenizerDecode(tokenizer_path, test_data_path);
  yllang::test::TestTokenizerBatchEncode(tokenizer_path, test_data_path);
  yllang::test::TestTokenizerBatchDecode(tokenizer_path, test_data_path);
  yllang::test::TestTokenizerEdgeCases(tokenizer_path);

  std::cout << "\nAll tests passed!\n";
  return 0;
}