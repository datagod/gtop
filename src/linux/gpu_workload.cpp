/* Copyright 2026 gtop++ contributors — Apache-2.0 (see NOTICE) */

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fmt/format.h>
#include <ranges>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "../btop_config.hpp"
#include "../btop_draw.hpp"
#include "../btop_log.hpp"
#include "../btop_shared.hpp"
#include "../btop_theme.hpp"
#include "../btop_tools.hpp"

namespace fs = std::filesystem;

using std::string;
using std::vector;
using namespace Tools;

#if defined(GPU_SUPPORT) && defined(__linux__)

namespace Gpu::Workload {

	int x = 1, y = 1, width = 0, height = 0;
	bool shown = false, redraw = true;
	string box;
	vector<gpu_proc_entry> entries;
	int last_reserved = 0;

	namespace {

	bool run_command(const string& cmd, string& out) {
		out.clear();
		FILE* pipe = popen(cmd.c_str(), "r");
		if (pipe == nullptr) return false;
		std::array<char, 4096> buf{};
		while (fgets(buf.data(), buf.size(), pipe) != nullptr)
			out += buf.data();
		const int status = pclose(pipe);
		return status == 0 and not out.empty();
	}

	string get_cmdline(unsigned int pid) {
		const auto raw = readfile(fs::path("/proc") / std::to_string(pid) / "cmdline");
		if (raw.empty()) return "";
		string cmd;
		cmd.reserve(raw.size());
		for (char c : raw) cmd += (c == '\0' ? ' ' : c);
		return string(trim(cmd));
	}

	string infer_label(const string& proc_name, const string& cmd) {
		static const std::regex model_flag(R"(--model\s+(\S+))", std::regex::icase);
		static const std::regex model_name(
			R"((qwen[\w.:-]+|llama[\w.:-]+|mistral[\w.:-]+|gemma[\w.:-]+|phi[\w.:-]+|deepseek[\w.:-]+))",
			std::regex::icase
		);

		if (std::smatch m; std::regex_search(cmd, m, model_flag)) {
			auto path = m[1].str();
			if (auto slash = path.find_last_of('/'); slash != string::npos)
				path = path.substr(slash + 1);
			return path.size() > 36 ? path.substr(0, 36) : path;
		}
		if (std::smatch m; std::regex_search(cmd, m, model_name))
			return m[1].str();

		if (proc_name.contains("llama-server"))
			return "llama-server";
		if (proc_name.contains("ollama"))
			return "ollama";
		if (proc_name.contains("vllm"))
			return "vllm";
		if (proc_name.contains("python") and (cmd.contains("chatterbox") or cmd.contains("tts")))
			return "TTS";
		if (cmd.contains("ffmpeg") or cmd.contains("nvenc"))
			return "video";

		return proc_name.empty() ? "unknown" : proc_name;
	}

	std::unordered_map<string, int> gpu_uuid_map() {
		std::unordered_map<string, int> map;
		string out;
		if (not run_command(
			"nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null",
			out
		)) return map;

		for (auto& line : ssplit(out, '\n')) {
			if (line.empty()) continue;
			const auto comma = line.find(',');
			if (comma == string::npos) continue;
			try {
				map[line.substr(comma + 1)] = std::stoi(string(trim(line.substr(0, comma))));
			} catch (...) {}
		}
		return map;
	}

	std::unordered_map<unsigned int, int> collect_pmon() {
		std::unordered_map<unsigned int, int> util;
		string out;
		if (not run_command("nvidia-smi pmon -c 1 2>/dev/null", out))
			return util;

		for (auto& line : ssplit(out, '\n')) {
			line = trim(line);
			if (line.empty() or line[0] == '#') continue;
			std::istringstream iss(line);
			string gpu_s, pid_s, type, sm_s;
			iss >> gpu_s >> pid_s >> type >> sm_s;
			if (sm_s == "-" or not isint(pid_s) or not isint(sm_s)) continue;
			util[std::stoul(pid_s)] = std::stoi(sm_s);
		}
		return util;
	}

	} // namespace

	void collect(const bool no_update) {
		if (no_update) return;

		entries.clear();
		if (not Config::getB("show_gpu_workloads") or Gpu::count < 1)
			return;

		const auto uuid_map = gpu_uuid_map();
		if (uuid_map.empty()) return;

		const auto pmon = collect_pmon();

		string out;
		if (not run_command(
			"nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory "
			"--format=csv,noheader,nounits 2>/dev/null",
			out
		)) return;

		for (auto& line : ssplit(out, '\n')) {
			line = trim(line);
			if (line.empty()) continue;

			const auto parts = ssplit(line, ',');
			if (parts.size() < 4) continue;

			const auto& uuid = parts[0];
			if (not uuid_map.contains(uuid)) continue;

			gpu_proc_entry e;
			e.gpu_index = uuid_map.at(uuid);
			try {
				e.pid = std::stoul(string(trim(parts[1])));
				e.process_name = string(trim(parts[2]));
				e.vram_bytes = std::stoll(string(trim(parts[3]))) * 1024 * 1024;
			} catch (...) {
				continue;
			}

			const auto cmd = get_cmdline(e.pid);
			e.label = infer_label(e.process_name, cmd);
			if (pmon.contains(e.pid))
				e.sm_util = pmon.at(e.pid);

			entries.push_back(std::move(e));
		}

		std::ranges::sort(entries, [](const auto& a, const auto& b) {
			return a.gpu_index != b.gpu_index ? a.gpu_index < b.gpu_index : a.vram_bytes > b.vram_bytes;
		});

		const int need = reserved_height();
		if (need != last_reserved) {
			last_reserved = need;
			Global::resized = true;
		}

		redraw = true;
	}

	int reserved_height() {
		if (not Config::getB("show_gpu_workloads") or Gpu::shown < 1)
			return 0;
		const int lines = std::max(Gpu::count, static_cast<int>(entries.size()));
		return std::clamp(lines + 3, 5, 12);
	}

	string draw(const bool force_redraw, const bool data_same) {
		if (Runner::stopping or not shown) return "";
		if (force_redraw) redraw = true;

		string out;
		out.reserve(width * height);

		const int label_w = std::max(12, width - 34);
		const int max_rows = std::max(0, height - 3);

		if (redraw) {
			out += box;
			out += Mv::to(y + 1, x + 1) + Theme::c("hi_fg") + Fx::b
				+ ljust("GPU", 4) + ' '
				+ ljust("Model / process", label_w) + ' '
				+ rjust("VRAM", 7) + ' '
				+ rjust("SM%", 4) + ' '
				+ rjust("PID", 7)
				+ Fx::ub;
		}

		std::unordered_map<int, vector<const gpu_proc_entry*>> by_gpu;
		for (const auto& e : entries)
			by_gpu[e.gpu_index].push_back(&e);

		struct row_line { string text; bool idle = false; };
		vector<row_line> rows;
		rows.reserve(entries.size() + Gpu::count);

		const int per_gpu_max = std::max(1, max_rows / std::max(1, Gpu::count));

		for (int gpu = 0; gpu < Gpu::count; ++gpu) {
			if (not by_gpu.contains(gpu)) {
				rows.push_back({
					ljust("GPU" + to_string(gpu), 4) + ' '
						+ ljust("(idle)", label_w) + ' '
						+ rjust("-", 7) + ' '
						+ rjust("-", 4) + ' '
						+ rjust("-", 7),
					true
				});
				continue;
			}

			auto& procs = by_gpu[gpu];
			std::ranges::sort(procs, [](const auto* a, const auto* b) {
				return a->vram_bytes > b->vram_bytes;
			});
			if (static_cast<int>(procs.size()) > per_gpu_max)
				procs.resize(per_gpu_max);

			bool first = true;
			for (const auto* e : procs) {
				const string gpu_col = first ? ljust("GPU" + to_string(gpu), 4) : ljust("", 4);
				first = false;

				const string vram = floating_humanizer(e->vram_bytes);
				const string sm = e->sm_util >= 0 ? to_string(e->sm_util) + '%' : "-";
				const string label = uresize(e->label, label_w);

				rows.push_back({
					gpu_col + ' '
						+ ljust(label, label_w) + ' '
						+ rjust(vram, 7) + ' '
						+ rjust(sm, 4) + ' '
						+ rjust(to_string(e->pid), 7)
				});
			}
		}

		if (rows.size() > static_cast<size_t>(max_rows))
			rows.resize(max_rows);

		(void)data_same;

		int row = 2;
		for (const auto& line : rows) {
			out += Mv::to(y + row++, x + 1)
				+ (line.idle ? Theme::c("inactive_fg") : Theme::c("main_fg"))
				+ line.text;
		}

		redraw = false;
		return out + Fx::reset;
	}

} // namespace Gpu::Workload

#elif defined(GPU_SUPPORT)

namespace Gpu::Workload {
	int x = 1, y = 1, width = 0, height = 0;
	bool shown = false, redraw = true;
	string box;
	vector<gpu_proc_entry> entries;

	void collect(const bool) {}
	string draw(const bool, const bool) { return ""; }
	int reserved_height() { return 0; }
}

#endif