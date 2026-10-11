from track.run_yolo import _parse_args, main

if __name__ == "__main__":
    args = _parse_args()
    main(
        source=args.source,
        conf=args.conf,
        iou_thresh=args.iou,
        max_age=args.max_age,
        min_hits=args.min_hits,
    )