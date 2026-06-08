/* Copyright 2026 gtop++ contributors — Apache-2.0 (see NOTICE) */

#include <algorithm>
#include <array>
#include <climits>
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

	string basename_of(const string& path) {
		if (auto slash = path.find_last_of('/'); slash != string::npos)
			return path.substr(slash + 1);
		return path;
	}

	string infer_label(const string& proc_name, const string& cmd) {
		static const std::regex model_flag(R"((?:--model|-m)\s+(\S+))", std::regex::icase);
		static const std::regex model_name(
			R"((qwen[\w.:-]+|llama[\w.:-]+|mistral[\w.:-]+|gemma[\w.:-]+|phi[\w.:-]+|deepseek[\w.:-]+|codellama[\w.:-]+))",
			std::regex::icase
		);
		static const std::regex ollama_model(R"((?:OLLAMA_MODEL|MODEL)=([^\s]+))", std::regex::icase);

		if (std::smatch m; std::regex_search(cmd, m, model_flag)) {
			auto path = m[1].str();
			if (path.contains("/blobs/") or path.contains("sha256-"))
				return "ollama model";
			path = basename_of(path);
			return path.size() > 48 ? path.substr(0, 48) : path;
		}
		if (std::smatch m; std::regex_search(cmd, m, model_name))
			return m[1].str();
		if (std::smatch m; std::regex_search(cmd, m, ollama_model))
			return basename_of(m[1].str());

		if (proc_name.contains("llama-server"))
			return "llama-server";
		if (proc_name.contains("ollama"))
			return "ollama";
		if (proc_name.contains("vllm"))
			return "vllm";
		if (proc_name.contains("frigate"))
			return basename_of(proc_name);
		if (proc_name.contains("python") and (cmd.contains("chatterbox") or cmd.contains("tts")))
			return "TTS";
		if (cmd.contains("ffmpeg") or cmd.contains("nvenc"))
			return "video";

		const auto base = basename_of(proc_name);
		return base.empty() ? "unknown" : base;
	}

	string classify_category(const string& proc_name, const string& cmd) {
		const string blob = proc_name + " " + cmd;
		static const std::regex llm(
			R"(llama-server|llama\.cpp|vllm|text-generation|ollama|exllama|sglang|tritonserver)",
			std::regex::icase
		);
		static const std::regex tts(R"(chatterbox|tts|bark|piper|coqui)", std::regex::icase);
		static const std::regex vision(R"(frigate|yolo|detect|embeddings|onnxruntime|tensorrt)", std::regex::icase);
		static const std::regex video(R"(ffmpeg|gstreamer|nvenc|nvdec)", std::regex::icase);
		static const std::regex embed(R"(embed|sentence-transformers|tei)", std::regex::icase);
		static const std::regex diffusion(R"(comfyui|stable.?diff|diffusers)", std::regex::icase);

		if (std::regex_search(blob, llm)) return "LLM";
		if (std::regex_search(blob, tts)) return "TTS";
		if (std::regex_search(blob, vision)) return "Vision";
		if (std::regex_search(blob, video)) return "Video";
		if (std::regex_search(blob, embed)) return "Embed";
		if (std::regex_search(blob, diffusion)) return "Image";
		return "";
	}

	struct ollama_running {
		string name;
		long long size_vram = 0;
	};

	vector<ollama_running> parse_ollama_ps_json(const string& json) {
		vector<ollama_running> models;
		size_t pos = 0;
		while ((pos = json.find("\"name\"", pos)) != string::npos) {
			pos += 6;
			const auto q1 = json.find('"', pos);
			if (q1 == string::npos) break;
			const auto q2 = json.find('"', q1 + 1);
			if (q2 == string::npos) break;
			const string name = json.substr(q1 + 1, q2 - q1 - 1);

			long long size_vram = 0;
			if (const auto vpos = json.find("\"size_vram\"", q2); vpos != string::npos and vpos < q2 + 512) {
				const auto num_start = json.find_first_of("0123456789", vpos);
				if (num_start != string::npos) {
					try { size_vram = std::stoll(json.substr(num_start)); } catch (...) {}
				}
			}

			if (not name.empty()) {
				bool duplicate = false;
				for (const auto& existing : models) {
					if (existing.name == name) {
						duplicate = true;
						break;
					}
				}
				if (not duplicate)
					models.push_back({name, size_vram});
			}
			pos = q2 + 1;
		}
		return models;
	}

	vector<ollama_running> collect_ollama_running() {
		vector<ollama_running> models;
		string out;

		if (run_command("curl -sf --max-time 2 http://127.0.0.1:11434/api/ps 2>/dev/null", out)) {
			models = parse_ollama_ps_json(out);
			if (not models.empty()) return models;
		}

		if (run_command("docker ps --format '{{.Names}}' 2>/dev/null", out)) {
			for (auto& cname : ssplit(out, '\n')) {
				cname = trim(cname);
				if (cname.empty() or (not cname.contains("ollama") and cname != "ollama"))
					continue;
				string api_out;
				if (run_command(
					"docker exec " + cname + " curl -sf --max-time 2 http://127.0.0.1:11434/api/ps 2>/dev/null",
					api_out
				)) {
					models = parse_ollama_ps_json(api_out);
					if (not models.empty()) return models;
				}
			}
		}

		auto parse_ollama_ps_text = [](const string& text) {
			vector<ollama_running> parsed;
			for (auto& line : ssplit(text, '\n')) {
				line = trim(line);
				if (line.empty() or line.starts_with("NAME"))
					continue;
				const auto parts = ssplit(line, ' ');
				if (not parts.empty())
					parsed.push_back({string(parts[0]), 0});
			}
			return parsed;
		};

		if (run_command("docker exec ollama ollama ps 2>/dev/null", out))
			return parse_ollama_ps_text(out);
		if (run_command("ollama ps 2>/dev/null", out))
			return parse_ollama_ps_text(out);

		return models;
	}

	string match_ollama_model(const long long vram_bytes, const vector<ollama_running>& models) {
		if (models.empty()) return "";
		if (models.size() == 1) return models[0].name;

		if (vram_bytes > 0) {
			long long best_diff = LLONG_MAX;
			string best;
			for (const auto& m : models) {
				if (m.size_vram <= 0) continue;
				const long long diff = llabs(m.size_vram - vram_bytes);
				if (diff < best_diff) {
					best_diff = diff;
					best = m.name;
				}
			}
			if (not best.empty() and best_diff <= vram_bytes / 5 + 256LL * 1024 * 1024)
				return best;
		}

		for (const auto& m : models)
			if (not m.name.empty()) return m.name;
		return "";
	}

	string resolve_llm_model(
		const string& proc_name,
		const string& cmd,
		const long long vram_bytes,
		const vector<ollama_running>& ollama_models
	) {
		static const std::regex served_model(R"(--served-model-name\s+(\S+))", std::regex::icase);
		static const std::regex model_path(R"((?:--model-path|--model|-m)\s+(\S+))", std::regex::icase);
		static const std::regex model_id(R"(--model-id\s+(\S+))", std::regex::icase);
		static const std::regex model_name_re(
			R"((qwen[\w.:-]+|llama[\w.:-]+|mistral[\w.:-]+|gemma[\w.:-]+|phi[\w.:-]+|deepseek[\w.:-]+|codellama[\w.:-]+|moondream[\w.:-]+))",
			std::regex::icase
		);

		if (std::smatch m; std::regex_search(cmd, m, served_model))
			return m[1].str();
		if (std::smatch m; std::regex_search(cmd, m, model_id))
			return basename_of(m[1].str());

		if (std::smatch m; std::regex_search(cmd, m, model_path)) {
			const auto path = m[1].str();
			if (not path.contains("/blobs/") and not path.contains("sha256-"))
				return basename_of(path);
		}

		if (proc_name.contains("llama-server") or proc_name.contains("ollama") or cmd.contains("/ollama/")) {
			if (const auto ollama = match_ollama_model(vram_bytes, ollama_models); not ollama.empty())
				return ollama;
		}

		if (std::smatch m; std::regex_search(cmd, m, model_name_re))
			return m[1].str();

		return "";
	}

	string cmd_context(const string& cmd) {
		if (cmd.empty()) return "";

		static const std::regex py_script(R"(\bpython\d*(?:\.\d+)?\s+(\S+))", std::regex::icase);
		if (std::smatch m; std::regex_search(cmd, m, py_script))
			return basename_of(m[1].str());

		static const std::regex serve_arg(R"(serve\s+(\S+))", std::regex::icase);
		if (std::smatch m; std::regex_search(cmd, m, serve_arg))
			return basename_of(m[1].str());

		if (cmd.contains("ffmpeg")) {
			static const std::regex stream(R"(-i\s+(\S+))");
			if (std::smatch m; std::regex_search(cmd, m, stream))
				return "stream " + basename_of(m[1].str());
			return "ffmpeg transcode";
		}

		return "";
	}

	std::unordered_map<string, string> docker_name_map() {
		std::unordered_map<string, string> map;
		string out;
		if (not run_command("docker ps --format '{{.ID}}\t{{.Names}}' 2>/dev/null", out))
			return map;

		for (auto& line : ssplit(out, '\n')) {
			line = trim(line);
			if (line.empty()) continue;
			const auto tab = line.find('\t');
			if (tab == string::npos) continue;
			const auto id = line.substr(0, tab);
			const auto name = line.substr(tab + 1);
			map[id] = name;
			if (id.size() >= 12)
				map[id.substr(0, 12)] = name;
		}
		return map;
	}

	string container_for_pid(const unsigned int pid, const std::unordered_map<string, string>& docker_names) {
		const auto cgroup = readfile(fs::path("/proc") / std::to_string(pid) / "cgroup");
		if (cgroup.empty()) return "";

		static const std::regex docker_scope(R"(docker[/-]([a-f0-9]{12,64}))", std::regex::icase);
		if (std::smatch m; std::regex_search(cgroup, m, docker_scope)) {
			const auto& id = m[1].str();
			if (docker_names.contains(id))
				return docker_names.at(id);
			if (id.size() >= 12 and docker_names.contains(id.substr(0, 12)))
				return docker_names.at(id.substr(0, 12));
		}
		return "";
	}

	string build_detail(
		const string& label,
		const string& proc_name,
		const string& cmd,
		const string& container
	) {
		string line = label;

		vector<string> parts;
		if (not container.empty())
			parts.push_back("docker:" + container);

		const auto proc_base = basename_of(proc_name);
		if (not proc_base.empty() and proc_base != label and not label.contains(proc_base))
			parts.push_back(proc_base);

		const auto ctx = cmd_context(cmd);
		if (not ctx.empty() and ctx != label and not v_contains(parts, ctx))
			parts.push_back(ctx);

		if (parts.empty() and not cmd.empty() and cmd != proc_name) {
			auto snippet = cmd;
			if (snippet.size() > 72)
				snippet = snippet.substr(0, 69) + "...";
			if (snippet != label)
				parts.push_back(snippet);
		}

		for (size_t i = 0; i < parts.size(); ++i) {
			line += (i == 0 ? " · " : " · ") + parts[i];
		}

		return line;
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
				map[string(trim(line.substr(comma + 1)))] = std::stoi(string(trim(line.substr(0, comma))));
			} catch (...) {}
		}
		return map;
	}

	struct pmon_stats {
		int sm_util = -1;
		int enc_util = -1;
		int dec_util = -1;
	};

	int parse_pmon_pct(const string& val) {
		return (val != "-" and isint(val)) ? std::stoi(val) : -1;
	}

	std::unordered_map<unsigned int, pmon_stats> collect_pmon() {
		std::unordered_map<unsigned int, pmon_stats> util;
		string out;
		if (not run_command("nvidia-smi pmon -c 1 2>/dev/null", out))
			return util;

		for (auto& line : ssplit(out, '\n')) {
			line = trim(line);
			if (line.empty() or line[0] == '#') continue;
			std::istringstream iss(line);
			string gpu_s, pid_s, type, sm_s, mem_s, enc_s, dec_s;
			iss >> gpu_s >> pid_s >> type >> sm_s >> mem_s >> enc_s >> dec_s;
			if (not isint(pid_s)) continue;

			pmon_stats stats;
			stats.sm_util = parse_pmon_pct(sm_s);
			stats.enc_util = parse_pmon_pct(enc_s);
			stats.dec_util = parse_pmon_pct(dec_s);
			util[std::stoul(pid_s)] = stats;
		}
		return util;
	}

	int get_process_runtime_s(const unsigned int pid) {
		const auto stat = readfile(fs::path("/proc") / std::to_string(pid) / "stat");
		if (stat.empty()) return -1;

		const auto rparen = stat.rfind(')');
		if (rparen == string::npos) return -1;

		std::istringstream iss(stat.substr(rparen + 2));
		string token;
		for (int field = 3; field <= 22; ++field) {
			if (not (iss >> token)) return -1;
			if (field == 22) {
				try {
					const unsigned long long start = std::stoull(token);
					const auto uptime_raw = readfile(fs::path("/proc/uptime"));
					if (uptime_raw.empty()) return -1;
					const double uptime = std::stod(string(trim(uptime_raw.substr(0, uptime_raw.find(' ')))));
					const long hz = sysconf(_SC_CLK_TCK);
					return std::max(0, static_cast<int>(uptime - start / (hz > 0 ? hz : 100)));
				} catch (...) {
					return -1;
				}
			}
		}
		return -1;
	}

	string format_runtime(const int seconds) {
		if (seconds < 0) return "-";
		if (seconds < 60) return to_string(seconds) + 's';
		if (seconds < 3600) return to_string(seconds / 60) + 'm';
		return to_string(seconds / 3600) + 'h' + to_string((seconds % 3600) / 60) + 'm';
	}

	} // namespace

	void collect(const bool no_update) {
		(void)no_update;

		entries.clear();
		if (not Config::getB("show_gpu_workloads") or Gpu::count < 1)
			return;

		const auto uuid_map = gpu_uuid_map();
		if (uuid_map.empty()) return;

		const auto pmon = collect_pmon();
		const auto docker_names = docker_name_map();
		const auto ollama_running = collect_ollama_running();

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

			const string uuid = string(trim(parts[0]));
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
			const auto container = container_for_pid(e.pid, docker_names);
			e.category = classify_category(e.process_name, cmd + " " + container);
			if (e.category == "LLM" or e.process_name.contains("llama-server") or e.process_name.contains("vllm")) {
				if (const auto model = resolve_llm_model(e.process_name, cmd, e.vram_bytes, ollama_running);
					not model.empty())
					e.label = model;
			}
			e.detail = build_detail(e.label, e.process_name, cmd, container);
			e.runtime_s = get_process_runtime_s(e.pid);
			if (pmon.contains(e.pid)) {
				const auto& stats = pmon.at(e.pid);
				e.sm_util = stats.sm_util;
				e.enc_util = stats.enc_util;
				e.dec_util = stats.dec_util;
			}

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

		constexpr int gpu_w = 4;
		constexpr int cat_w = 8;
		constexpr int vram_w = 7;
		constexpr int sm_w = 4;
		constexpr int uptime_w = 6;
		constexpr int enc_w = 4;
		constexpr int dec_w = 4;
		constexpr int col_gpu = 1;
		constexpr int col_type = col_gpu + gpu_w + 1;
		constexpr int col_work = col_type + cat_w + 1;
		constexpr int fixed_tail = 1 + vram_w + 1 + sm_w + 1 + uptime_w + 1 + enc_w + 1 + dec_w;
		const int detail_w = std::max(10, width - col_work - fixed_tail + 1);
		const int col_vram = width - (dec_w + 1 + enc_w + 1 + uptime_w + 1 + sm_w + 1 + vram_w) + 1;
		const int col_sm = col_vram + vram_w + 1;
		const int col_up = col_sm + sm_w + 1;
		const int col_enc = col_up + uptime_w + 1;
		const int col_dec = col_enc + enc_w + 1;
		const int max_rows = std::max(0, height - 3);

		const auto cell = [&](const int row_y, const int col_x, const string& content) {
			out += Mv::to(y + row_y, x + col_x);
			out += content;
		};

		const auto category_col = [&](const string& category) -> string {
			if (category.empty()) return ljust("-", cat_w, true);
			return ljust(uresize('[' + category + ']', cat_w), cat_w, true);
		};

		const auto pct_cell = [](const int val, const int w) -> string {
			if (val < 0)
				return Theme::c("inactive_fg") + rjust("-", w);
			const string text = to_string(val) + '%';
			return Theme::g("cpu").at(std::clamp(val, 0, 100)) + Fx::b + rjust(text, w) + Fx::ub;
		};

		if (redraw) {
			out += box;
			const string hdr = Theme::c("title") + Fx::b;
			cell(1, col_gpu, hdr + Theme::c("hi_fg") + ljust("GPU", gpu_w) + Fx::ub);
			cell(1, col_type, hdr + ljust("Type", cat_w));
			cell(1, col_work, hdr + ljust("Workload", detail_w));
			cell(1, col_vram, hdr + rjust("VRAM", vram_w));
			cell(1, col_sm, hdr + rjust("SM%", sm_w));
			cell(1, col_up, hdr + rjust("Up", uptime_w));
			cell(1, col_enc, hdr + rjust("ENC", enc_w));
			cell(1, col_dec, hdr + rjust("DEC", dec_w) + Fx::ub);
		}

		std::unordered_map<int, vector<const gpu_proc_entry*>> by_gpu;
		for (const auto& e : entries)
			by_gpu[e.gpu_index].push_back(&e);

		struct wl_row {
			int gpu = -1;
			bool show_gpu = false;
			bool idle = false;
			string category;
			string detail;
			long long vram_bytes = 0;
			int sm_util = -1;
			int enc_util = -1;
			int dec_util = -1;
			int runtime_s = -1;
		};
		vector<wl_row> rows;
		rows.reserve(entries.size() + Gpu::count);

		const int per_gpu_max = std::max(1, max_rows / std::max(1, Gpu::count));

		for (int gpu = 0; gpu < Gpu::count; ++gpu) {
			if (not by_gpu.contains(gpu)) {
				rows.push_back({.gpu = gpu, .show_gpu = true, .idle = true, .category = "", .detail = "(idle)"});
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
				rows.push_back({
					.gpu = gpu,
					.show_gpu = first,
					.category = e->category,
					.detail = e->detail,
					.vram_bytes = e->vram_bytes,
					.sm_util = e->sm_util,
					.enc_util = e->enc_util,
					.dec_util = e->dec_util,
					.runtime_s = e->runtime_s
				});
				first = false;
			}
		}

		if (rows.size() > static_cast<size_t>(max_rows))
			rows.resize(max_rows);

		(void)data_same;

		long long max_vram = 1;
		for (const auto& e : entries)
			max_vram = std::max(max_vram, e.vram_bytes);

		int row = 2;
		for (const auto& line : rows) {
			if (line.idle) {
				cell(row, col_gpu, Theme::c("hi_fg") + Fx::b + ljust("GPU" + to_string(line.gpu), gpu_w) + Fx::ub);
				cell(row, col_type, Theme::c("inactive_fg") + ljust("-", cat_w, true));
				cell(row, col_work, Theme::c("inactive_fg") + ljust(line.detail, detail_w, true));
				cell(row, col_vram, Theme::c("inactive_fg") + rjust("-", vram_w));
				cell(row, col_sm, Theme::c("inactive_fg") + rjust("-", sm_w));
				cell(row, col_up, Theme::c("inactive_fg") + rjust("-", uptime_w));
				cell(row, col_enc, Theme::c("inactive_fg") + rjust("-", enc_w));
				cell(row, col_dec, Theme::c("inactive_fg") + rjust("-", dec_w));
				++row;
				continue;
			}

			const string gpu_col = line.show_gpu ? ljust("GPU" + to_string(line.gpu), gpu_w) : ljust("", gpu_w);
			const string detail = ljust(uresize(line.detail, detail_w), detail_w, true);
			const string vram = floating_humanizer(line.vram_bytes);
			const int vram_pct = static_cast<int>(std::clamp(line.vram_bytes * 100 / max_vram, 0LL, 100LL));
			const string sm = line.sm_util >= 0 ? to_string(line.sm_util) + '%' : "-";
			const string uptime = rjust(uresize(format_runtime(line.runtime_s), uptime_w), uptime_w);

			cell(row, col_gpu, Theme::c("hi_fg") + Fx::b + gpu_col + Fx::ub);
			cell(row, col_type, Theme::c("proc_misc") + Fx::b + category_col(line.category) + Fx::ub);
			cell(row, col_work, Theme::c("title") + Fx::b + detail + Fx::ub);
			cell(row, col_vram, Theme::g("used").at(vram_pct) + Fx::b + rjust(vram, vram_w) + Fx::ub);
			cell(row, col_sm, line.sm_util >= 0
				? Theme::g("cpu").at(std::clamp(line.sm_util, 0, 100)) + Fx::b + rjust(sm, sm_w) + Fx::ub
				: Theme::c("inactive_fg") + rjust(sm, sm_w));
			cell(row, col_up, Theme::c("title") + uptime);
			cell(row, col_enc, pct_cell(line.enc_util, enc_w));
			cell(row, col_dec, pct_cell(line.dec_util, dec_w));
			++row;
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