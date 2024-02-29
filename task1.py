from util import *
def task1(dataset):

    item_count = 7
    min_supports = [0.01,0.03,0.05,0.10,0.15,0.20]

    apriori_itemsets_collection = []
    eclat_itemsets_collection = []
    fpgrowth_itemsets_collection = []

    times_apriori = []
    times_eclat = []
    times_fpgrowth = []
    for min_support in min_supports:
        print(f'experiment for support {min_support}')
        itemsets_apriori, time_apriori = apriori_analyze(dataset, min_support, item_count)
        itemsets_eclat, time_eclat = eclat_analyze(dataset, min_support, item_count)
        itemsets_fpgrowth, time_fpgrowth = fpgrowth_analyze(dataset, min_support, item_count)

        apriori_itemsets_collection.append(itemsets_apriori)
        eclat_itemsets_collection.append(itemsets_eclat)
        fpgrowth_itemsets_collection.append(itemsets_fpgrowth)

        times_apriori.append(time_apriori)
        times_eclat.append(time_eclat)
        times_fpgrowth.append(time_fpgrowth)

        #visualize_items_count_per_support
        #visualize_maxlen_per_support
        #visualize_items_count_of_different_lengths_per_support

    #сравнение быстродействия алгоритмов на фиксированном наборе данных при изменяемом пороге поддержки;
    visualize_results(
        items=[times_apriori,times_eclat,times_fpgrowth],
        supports=min_supports,
        title='Быстродействие алгоритмов при разной поддержке',
        legend=['apriori','eclat','fpgrowth'],
        xlabel='Поддержка',
        ylabel='Быстродействие, с',
        savename='Быстродействие'
        )

    # общее количество частых наборов объектов на фиксированном наборе данных при изменяемом пороге поддержки;
    counts_apriori = [len(item) for item in apriori_itemsets_collection]
    counts_eclat = [len(item) for item in eclat_itemsets_collection]
    counts_fpgrowth = [len(item) for item in fpgrowth_itemsets_collection]
    visualize_results(
        items=[counts_apriori, counts_eclat, counts_fpgrowth],
        supports=min_supports,
        title='Количество частых наборов при разной поддержке',
        legend=['apriori', 'eclat', 'fpgrowth'],
        xlabel='Поддержка',
        ylabel='Количество',
        savename='ОбщееКоличество'
    )


    # максимальная длина частого набора объектов на фиксированном наборе данных при изменяемом пороге поддержки;
    itemlen_apriori = [item['itemsets'].apply(len).max() for item in apriori_itemsets_collection]
    itemlen_eclat = [item['itemsets'].apply(len).max() for item in eclat_itemsets_collection]
    itemlen_fpgrowth = [item['itemsets'].apply(len).max() for item in fpgrowth_itemsets_collection]
    visualize_results(
        items=[itemlen_apriori, itemlen_eclat, itemlen_fpgrowth],
        supports=min_supports,
        title='Макс. длина частого набора при разной поддержке',
        legend=['apriori', 'eclat', 'fpgrowth'],
        xlabel='Поддержка',
        ylabel='Длина',
        savename='МаксДлина'
    )

    # количество частых наборов объектов различной длины на фиксированном наборе данных при изменяемом пороге поддержки.
    for idx, min_support in enumerate(min_supports):
        x_apriori, y_apriori = get_lengths_and_counts(apriori_itemsets_collection[idx])
        x_eclat, y_eclat = get_lengths_and_counts(eclat_itemsets_collection[idx])
        x_fpgrowth, y_fpgrowth = get_lengths_and_counts(fpgrowth_itemsets_collection[idx])


        visualize_results(
            items=[y_apriori],
            supports=x_apriori,
            title=f'Количество частых наборов различной длины при поддержке {min_support}',
            legend=['apriori'],
            xlabel='Длина',
            ylabel='Количество',
            bar=True,
            colors = ['r'],
            savename=f'КолвоРазнойДлины_apriori_{min_support}'
        )
        visualize_results(
            items=[y_eclat],
            supports=x_eclat,
            title=f'Количество частых наборов различной длины при поддержке {min_support}',
            legend=['eclat'],
            xlabel='Длина',
            ylabel='Количество',
            bar=True,
            colors = ['g'],
            savename=f'КолвоРазнойДлины_eclat_{min_support}'
        )
        visualize_results(
            items=[y_fpgrowth],
            supports=x_fpgrowth,
            title=f'Количество частых наборов различной длины при поддержке {min_support}',
            legend=['fpgrowth'],
            xlabel='Длина',
            ylabel='Количество',
            bar=True,
            colors = ['b'],
            savename=f'КолвоРазнойДлины_fpgrowth_{min_support}'
        )