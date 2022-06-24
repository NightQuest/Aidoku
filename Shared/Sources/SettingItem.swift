//
//  SettingItem.swift
//  Aidoku
//
//  Created by Skitty on 2/17/22.
//

import Foundation

// types: group, select, multi-select, switch, stepper, segment, text, page, button, link
// TODO: slider
struct SettingItem: Codable {
    var type: String

    var key: String?
    var action: String?
    var title: String?
    var subtitle: String?
    var footer: String?
    var placeholder: String?
    var values: [String]?
    var titles: [String]?
    var defaultValue: JsonAnyValue?
    var notification: String?

    var requires: String?
    var requiresFalse: String?

    var authToEnable: Bool?
    var authToDisable: Bool?
    var authToOpen: Bool?

    // stepper
    var minimumValue: Double?
    var maximumValue: Double?

    var destructive: Bool? // button
    var external: Bool? // link

    var items: [SettingItem]? // group, page

    // text
    var autocapitalizationType: Int?
    var autocorrectionType: Int?
    var spellCheckingType: Int?
    var keyboardType: Int?
    var returnKeyType: Int?

    enum CodingKeys: String, CodingKey {
        case type
        case key
        case action
        case title
        case subtitle
        case footer
        case placeholder
        case values
        case titles
        case defaultValue = "default"
        case notification

        case requires
        case requiresFalse
        case authToEnable
        case authToDisable
        case authToOpen
        case minimumValue
        case maximumValue
        case destructive
        case external
        case items

        case autocapitalizationType
        case autocorrectionType
        case spellCheckingType
        case keyboardType
        case returnKeyType
    }
}
