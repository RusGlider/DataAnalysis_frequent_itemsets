from util import *
def task2(dataset):
    min_support = 0.1
    item_count = 7

    print(f'experiment for support {min_support}')
    itemsets_apriori, _ = apriori_analyze(dataset, min_support, item_count)
    itemsets_eclat, _ = eclat_analyze(dataset, min_support, item_count)
    itemsets_fpgrowth, _ = fpgrowth_analyze(dataset, min_support, item_count)

    #apriori_itemsets_collection.append(itemsets_apriori)
    #eclat_itemsets_collection.append(itemsets_eclat)
    #fpgrowth_itemsets_collection.append(itemsets_fpgrowth)




    apriori_itemsets_collection = []
    eclat_itemsets_collection = []
    fpgrowth_itemsets_collection = []

    times_apriori = []
    times_eclat = []
    times_fpgrowth = []

    confidences = [0.75, 0.80, 0.90, 0.95]
    for confidence in confidences:
        print(f'finding association rules for confidence {confidence}')
        rules_apriori, time_apriori = find_association_rules(itemsets_apriori,confidence)
        rules_eclat, time_eclat = find_association_rules(itemsets_eclat,confidence)
        rules_fpgrowth, time_fpgrowth = find_association_rules(itemsets_fpgrowth,confidence)

        times_apriori.append(time_apriori)
        times_eclat.append(time_eclat)
        times_fpgrowth.append(time_fpgrowth)

        apriori_itemsets_collection.append(rules_apriori)
        eclat_itemsets_collection.append(rules_eclat)
        fpgrowth_itemsets_collection.append(rules_fpgrowth)


        #TODO отрисовать таблицы
        draw_table_rules(rules_apriori, confidence, item_count, out_name=f'rules_apriori_{confidence}')
        draw_table_rules(rules_eclat, confidence, item_count, out_name=f'rules_eclat_{confidence}')
        draw_table_rules(rules_fpgrowth, confidence, item_count, out_name=f'rules_fpgrowth_{confidence}')


        """
        rules = rules[rules.apply(lambda row: len(row['antecedents']) + len(row['consequents']) <= item_count, axis=1)]

        rules.sort_values(by='confidence', ascending=False)
        #table = PrettyTable(['antecedent -> consequent','support','confidence'])
        table = PrettyTable(['antecedent','==>','consequent', 'support', 'confidence'])
        for idx, row in rules[['antecedents','consequents','support','confidence']].iterrows():
            table.add_row([f'{list(row["antecedents"])}',' ==> ', f'{list(row["consequents"])}', row['support'], row['confidence']])
        #    table.add_row([f'{list(row["antecedents"])} ==> {list(row["consequents"])}',row['support'], row['confidence']])
        print(f'table for confidence {confidence}')
        print(table)
        print()
        """

    print('visualization')
    #сравнение быстродействия поиска правил на фиксированном наборе данных при изменяемом пороге достоверности;
    visualize_results(
        items=[times_apriori,times_eclat,times_fpgrowth],
        supports=confidences,
        title='Быстродействие поиска правил при разной достоверности',
        legend=['apriori','eclat','fpgrowth'],
        xlabel='Достоверность',
        ylabel='Быстродействие, с',
        savename='ассоц_Быстродействие'
        )

    #общее количество найденных правил на фиксированном наборе данных при изменяемом пороге достоверности;
    counts_apriori = [len(item) for item in apriori_itemsets_collection]
    counts_eclat = [len(item) for item in eclat_itemsets_collection]
    counts_fpgrowth = [len(item) for item in fpgrowth_itemsets_collection]
    visualize_results(
        items=[counts_apriori, counts_eclat, counts_fpgrowth],
        supports=confidences,
        title='Количество правил при разной достоверности',
        legend=['apriori', 'eclat', 'fpgrowth'],
        xlabel='Достоверность',
        ylabel='Количество',
        savename='ассоц_ОбщееКоличество'
    )

    #максимальное количество объектов в правиле на фиксированном наборе данных при изменяемом пороге достоверности;

    #rules = rules[rules.apply(lambda row: len(row['antecedents']) + len(row['consequents']) <= item_count, axis=1)]

    itemlen_apriori = [max(item.apply(lambda row: len(row['antecedents']) + len(row['consequents']),axis=1).values) for item in apriori_itemsets_collection]
    itemlen_eclat = [max(item.apply(lambda row: len(row['antecedents']) + len(row['consequents']),axis=1).values) for item in eclat_itemsets_collection]
    itemlen_fpgrowth = [max(item.apply(lambda row: len(row['antecedents']) + len(row['consequents']),axis=1).values) for item in fpgrowth_itemsets_collection]
    visualize_results(
        items=[itemlen_apriori, itemlen_eclat, itemlen_fpgrowth],
        supports=confidences,
        title='Макс. количество объектов в правиле при разной достоверности',
        legend=['apriori', 'eclat', 'fpgrowth'],
        xlabel='Достоверность',
        ylabel='Длина',
        savename='ассоц_МаксДлина'
    )

    #количество правил, в которых антецедент и консеквент суммарно включают в себя не более семи объектов, на фиксированном наборе данных при изменяемом пороге достоверности.
    items_lt_item_count_apriori = [ len(item[item.apply(lambda row: len(row['antecedents']) + len(row['consequents']) <= item_count,axis=1)]) for item in apriori_itemsets_collection]
    items_lt_item_count_eclat = [ len(item[item.apply(lambda row: len(row['antecedents']) + len(row['consequents']) <= item_count,axis=1)]) for item in eclat_itemsets_collection]
    items_lt_item_count_fpgrowth = [ len(item[item.apply(lambda row: len(row['antecedents']) + len(row['consequents']) <= item_count,axis=1)]) for item in fpgrowth_itemsets_collection]

    visualize_results(
        items=[items_lt_item_count_apriori, items_lt_item_count_eclat, items_lt_item_count_fpgrowth],
        supports=confidences,
        title=f'Кол-во правил не более {item_count} объектов при разной достоверности',
        legend=['apriori', 'eclat', 'fpgrowth'],
        xlabel='Достоверность',
        ylabel='Длина',
        savename=f'ассоц_КолПравилДо{item_count}'
    )