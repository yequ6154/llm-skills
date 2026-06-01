import SwiftUI

@main
struct GirlfriendCameraApp: App {
    var body: some Scene {
        WindowGroup {
            CameraScreen()
                .preferredColorScheme(.dark)
                .statusBarHidden(true)
        }
    }
}
