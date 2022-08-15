//
//  CoreDataManager+Category.swift
//  Aidoku (iOS)
//
//  Created by Skitty on 8/11/22.
//

import CoreData

extension CoreDataManager {

    /// Remove all category objects.
    func clearCategories(context: NSManagedObjectContext? = nil) {
        clear(request: CategoryObject.fetchRequest(), context: context)
    }

    /// Get category object with title.
    func getCategory(title: String, context: NSManagedObjectContext? = nil) -> CategoryObject? {
        let context = context ?? self.context
        let request = CategoryObject.fetchRequest()
        request.predicate = NSPredicate(format: "title == %@", title)
        request.fetchLimit = 1
        return (try? context.fetch(request))?.first
    }

    /// Get all category objects.
    func getCategories(context: NSManagedObjectContext? = nil) -> [CategoryObject] {
        let context = context ?? self.context
        let request = CategoryObject.fetchRequest()
        let objects = try? context.fetch(request)
        return objects ?? []
    }

    /// Get category objects for a library manga.
    func getCategories(sourceId: String, mangaId: String, context: NSManagedObjectContext? = nil) -> [CategoryObject] {
        let libraryObject = getLibraryManga(sourceId: sourceId, mangaId: mangaId, context: context)
        return (libraryObject?.categories?.allObjects as? [CategoryObject]) ?? []
    }

    /// Get category objects for a library object.
    func getCategories(libraryManga: LibraryMangaObject) -> [CategoryObject] {
        (libraryManga.categories?.allObjects as? [CategoryObject]) ?? []
    }

    /// Create a category object.
    @discardableResult
    func createCategory(title: String, context: NSManagedObjectContext? = nil) -> CategoryObject? {
        let context = context ?? self.context

        // check if category exists
        let request = CategoryObject.fetchRequest()
        request.predicate = NSPredicate(format: "title == %@", title)
        request.fetchLimit = 1
        guard
            case let categoryCount = (try? context.count(for: request)) ?? 0,
            categoryCount == 0
        else { return nil }

        request.predicate = nil
        request.sortDescriptors = [NSSortDescriptor(key: "sort", ascending: false)]
        let lastCategoryIndex = (try? context.fetch(request))?.first?.sort ?? -1

        let categoryObject = CategoryObject(context: context)
        categoryObject.title = title
        categoryObject.sort = lastCategoryIndex + 1
        return categoryObject
    }

    /// Add categories to library manga.
    func addCategoriesToManga(sourceId: String, mangaId: String, categories: [String], context: NSManagedObjectContext? = nil) {
        let context = context ?? self.context

        let libraryObject = getLibraryManga(sourceId: sourceId, mangaId: mangaId, context: context)
        for category in categories {
            guard let categoryObject = getCategory(title: category, context: context) else { continue }
            libraryObject?.addToCategories(categoryObject)
        }
    }
}
