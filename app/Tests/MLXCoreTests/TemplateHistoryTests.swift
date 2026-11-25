import Combine
import XCTest
@testable import MLXCore

/// Template videos made on this Mac: kept with what made them, listed newest
/// first, opened again as a template, and deleted with every file they own.
@MainActor
final class TemplateHistoryTests: XCTestCase {
    private var dir: URL!

    override func setUp() {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("tpl-history-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDown() { try? FileManager.default.removeItem(at: dir) }

    private func file(_ name: String) -> String {
        let url = dir.appendingPathComponent(name)
        FileManager.default.createFile(atPath: url.path, contents: Data("x".utf8))
        return url.path
    }

    private func entry(_ name: String, captioned: Bool = true, cast: [String: UUID] = [:]) -> TemplateHistoryEntry {
        TemplateHistoryEntry(template: MediaTemplate.pets.rawValue, clip: file("\(name).mp4"),
                             captioned: captioned ? file("\(name).captioned.mp4") : nil,
                             cast: cast, answers: ["topic": "socks"], lines: ["a": "Hi."],
                             shape: ShareShape.square.rawValue, seed: 7)
    }

    func testAVideoSurvivesARelaunchNewestFirst() {
        let history = TemplateHistory(rootDir: dir.appendingPathComponent("store"))
        history.add(entry("old"))
        history.add(entry("new"))
        let reopened = TemplateHistory(rootDir: dir.appendingPathComponent("store"))
        XCTAssertEqual(reopened.entries.map { URL(fileURLWithPath: $0.clip).lastPathComponent }, ["new.mp4", "old.mp4"])
    }

    func testAVideoWhoseFileIsGoneIsNotListedAndAMissingCaptionFallsBackToTheClip() throws {
        let history = TemplateHistory(rootDir: dir.appendingPathComponent("store"))
        let gone = entry("gone")
        let bare = entry("bare")
        history.add(gone)
        history.add(bare)
        try FileManager.default.removeItem(atPath: gone.clip)
        try FileManager.default.removeItem(atPath: try XCTUnwrap(bare.captioned))
        let reopened = TemplateHistory(rootDir: dir.appendingPathComponent("store"))
        XCTAssertEqual(reopened.entries.map(\.id), [bare.id])
        XCTAssertEqual(reopened.entries.first?.playURL.path, bare.clip)
    }

    func testCaptionsAddedLaterSurviveARelaunch() {
        let history = TemplateHistory(rootDir: dir.appendingPathComponent("store"))
        let e = entry("late", captioned: false)
        history.add(e)
        history.setCaptioned(e.id, file("late.captioned.mp4"))
        XCTAssertEqual(TemplateHistory(rootDir: dir.appendingPathComponent("store")).entries.first?.playURL.lastPathComponent,
                       "late.captioned.mp4")
    }

    func testCaptionRetryWaitsForTheCancelledJobToFinish() async {
        _ = NSApplication.shared
        let history = TemplateHistory(rootDir: dir.appendingPathComponent("store"))
        let saved = TemplateHistoryEntry(template: MediaTemplate.reaction.rawValue, clip: file("retry.mp4"),
                                        cast: [:], answers: [:], lines: [:], shape: "square", seed: 7)
        history.add(saved)
        let app = AppState(startServerAtLaunch: false)
        let run = TemplateRunner(history: history)
        run.restore(saved, cast: app.cast)
        let finished = expectation(description: "cancelled caption job returned")
        let completion = run.$working.dropFirst().filter { !$0 }.prefix(1).sink { _ in finished.fulfill() }

        run.addCaptions(app: app)
        XCTAssertEqual(run.phase, .captioning)
        run.cancel()
        XCTAssertEqual(run.phase, .done)
        XCTAssertTrue(run.working)
        run.addCaptions(app: app)
        XCTAssertEqual(run.phase, .done, "a cancelled job still owns the output until its task returns")

        await fulfillment(of: [finished], timeout: 2)
        withExtendedLifetime(completion) {}
        XCTAssertFalse(run.isBusy)
        run.addCaptions(app: app)
        XCTAssertEqual(run.phase, .captioning, "captions can be retried after the previous task returns")
        run.cancel()
    }

    func testDeletingAVideoRemovesItsClipCaptionsGIFAndSettingsSidecar() {
        let history = TemplateHistory(rootDir: dir.appendingPathComponent("store"))
        var e = entry("doomed")
        e.gif = file("doomed.gif")
        let sidecar = VideoGenService.sidecarPath(forVideo: e.clip)
        FileManager.default.createFile(atPath: sidecar, contents: Data("settings".utf8))
        history.add(e)
        history.delete(e.id)
        for path in [e.clip, e.captioned!, e.gif!, sidecar] {
            XCTAssertFalse(FileManager.default.fileExists(atPath: path), "\(path) was left behind")
        }
        XCTAssertTrue(TemplateHistory(rootDir: dir.appendingPathComponent("store")).entries.isEmpty)
    }

    func testOpeningAVideoAgainRestoresItsTemplateWithTheCastStillHere() throws {
        let cast = CastStore(rootDir: dir.appendingPathComponent("cast"))
        let photo = dir.appendingPathComponent("p.png")
        let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 8, pixelsHigh: 8, bitsPerSample: 8,
                                   samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                                   bytesPerRow: 0, bitsPerPixel: 0)!
        try rep.representation(using: .png, properties: [:])!.write(to: photo)
        let max = try cast.add(CastMember(name: "Max", kind: .dog, voice: .goofy, photo: ""), photo: photo)
        let e = entry("again", cast: ["a": max.id, "b": UUID()])
        let run = TemplateRunner(history: TemplateHistory(rootDir: dir.appendingPathComponent("store")))
        run.restore(e, cast: cast)
        XCTAssertEqual(run.template, .pets)
        XCTAssertEqual(run.fill.cast["a"], max)
        XCTAssertNil(run.fill.cast["b"], "a member deleted since is left out, not invented")
        XCTAssertEqual(run.lines, ["a": "Hi."])
        XCTAssertEqual(run.shape, .square)
        XCTAssertEqual(run.phase, .done)
        XCTAssertFalse(run.onGallery)
        XCTAssertEqual(run.playable?.path, e.captioned)
        run.showsCaptions = false
        XCTAssertEqual(run.playable?.path, e.clip)
    }

    /// The stage plays only a finished run: the clean clip used to start under
    /// the caption burn, and the captioned one started again after it.
    func testNothingPlaysBeforeTheRunIsDone() {
        let clip = URL(fileURLWithPath: "/clip.mp4")
        for phase: TemplateRunner.Phase in [.filling, .writing, .review, .rendering, .captioning, .failed("x")] {
            XCTAssertNil(TemplateRunner.playable(phase: phase, result: clip), "\(phase) must not play")
        }
        XCTAssertEqual(TemplateRunner.playable(phase: .done, result: clip), clip)
    }
}
