
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace hercules::common {

//
// An ASCII table printer.
//
class TablePrinter {
 public:
  // Insert a row at the end of the table
  void InsertRow(const std::vector<std::string>& row);

  // Print the table
  std::string PrintTable();

  // TablePrinter will take the ownership of `headers`.
  TablePrinter(const std::vector<std::string>& headers);

 private:
  // Update the `shares_` such that all the excess
  // amount of space not used a column is fairly allocated
  // to the other columns
  void FairShare();

  // Append a row to `table`. This function handles the cases where a wrapping
  // occurs.
  void AddRow(std::stringstream& table, size_t row_index);

  // Add a row divider
  void AddRowDivider(std::stringstream& table);

  // Max row width
  std::vector<size_t> max_widths_;

  // Max row height
  std::vector<size_t> max_heights_;

  // A vector of vectors of vectors containing data items for every column
  // The record is stored in a vector of string, where each of the vector items
  // contains a single line from the record. For example, ["Item 1", "Item 2",
  // "Item 3\n Item 3 line 2"] will be stored as [["Item 1"], ["Item 2"], ["Item
  // 3", "Item 3 line 2"]]
  std::vector<std::vector<std::vector<std::string>>> data_;

  // Fair share of every column
  std::vector<float> shares_;
};

}  // namespace hercules::common
