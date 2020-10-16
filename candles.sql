SELECT `_30m`,sum_price,sum_qty,cnt,mean_price,cost_sum,max_price,min_price,
		avg_price,qty_buy,qty_sell,open_price,close_price
FROM 
(
	SELECT `_30m`,sum_price,sum_qty,cnt,mean_price,cost_sum,max_price,min_price,
		avg_price,qty_buy,qty_sell,max_ts,open_price
	FROM (
		SELECT `_30m`,sum(price) AS sum_price,sum(qty) AS sum_qty,count(*) AS cnt,
			avg(price) AS mean_price,sum(price*qty) AS cost_sum,
			max(price) AS max_price,min(price) AS min_price,
			(cost_sum/sum_qty) AS avg_price,
			SUM(qty_buy) AS qty_buy,SUM(qty_sell) AS qty_sell,
			max(_timestamp) AS max_ts,min(_timestamp) AS min_ts
			
		FROM (
			SELECT `_timestamp`,price,qty,cost,'buy' AS _type,

				floor((`_timestamp`- ({ts_now}-({steps}*{granule})))/{granule}) AS _30m,
				qty AS qty_buy,0 AS qty_sell
			FROM {asset}.tick_buy
			WHERE `_timestamp` >= ({ts_now} - {steps}*{granule}) AND `_timestamp` < {ts_now}
			UNION ALL 
			SELECT `_timestamp`,price,qty,cost,'sell' AS _type,
				floor((`_timestamp`- ({ts_now}-({steps}*{granule})))/{granule}) AS _30m,
				0 AS qty_buy,qty AS qty_sell
			FROM {asset}.tick_sell
			WHERE `_timestamp` >= ({ts_now} - {steps}*{granule}) AND `_timestamp` < {ts_now}
		)
		GROUP BY `_30m`
	)ALL INNER JOIN (
		SELECT min_ts, SUM(p*q)/SUM(q) AS open_price
		FROM (
			SELECT _timestamp AS min_ts, price AS p,qty AS q
			FROM {asset}.tick_buy
			WHERE `_timestamp` >= ({ts_now} - {steps}*{granule}) AND `_timestamp` < {ts_now}
			UNION ALL 
			SELECT _timestamp, price AS p,qty AS q
			FROM {asset}.tick_sell
			WHERE `_timestamp` >= ({ts_now} - {steps}*{granule}) AND `_timestamp` < {ts_now}
		)
		GROUP BY min_ts
	)USING (min_ts)
)ALL INNER JOIN (
	SELECT max_ts, SUM(p*q)/SUM(q) AS close_price
	FROM (
		SELECT _timestamp AS max_ts, price AS p,qty AS q
		FROM {asset}.tick_buy
		WHERE `_timestamp` >= ({ts_now} - {steps}*{granule}) AND `_timestamp` < {ts_now}
		UNION ALL 
		SELECT _timestamp, price AS p,qty AS q
		FROM {asset}.tick_sell
		WHERE `_timestamp` >= ({ts_now} - {steps}*{granule}) AND `_timestamp` < {ts_now}
	)
	GROUP BY max_ts
)USING (max_ts)
ORDER BY `_30m`